import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


# Loads a CSV file into a pandas DataFrame
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Displays the number of rows and columns in the dataset inside a colored box
def display_data_shape(df, div_color):
    rows_count, columns_count = df.shape
    st.markdown(f"""
        <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                The dataset contains:
                <ul>
                 <li><strong>Rows:</strong> {rows_count}</li>
                 <li><strong>Columns:</strong> {columns_count}</li>
                </ul>
        </div>
        <hr>
    """, unsafe_allow_html=True)

# Creates and displays a resized Seaborn pairplot for selected columns
def plot_pairplot(df, cols, title_color):
    # add a title before the plot
    st.markdown(f"<h2 style='color: {title_color};'>Pairplot of {', '.join(cols)}</h2>", unsafe_allow_html=True)
    # Create the pairplot
    pairplot_fig = sns.pairplot(df[cols])
    # Resize the pairplot to make it smaller
    pairplot_fig.fig.set_size_inches(4, 4)
    # Display the plot in Streamlit
    st.pyplot(pairplot_fig, use_container_width=False)

# Displays a scatter plot of clusters based on two features
def plot_clusters(df, x_col, y_col):
    # add subheader
    st.subheader("Cluster Visualization")
    # create a figure
    fig, ax = plt.subplots(figsize=(4, 2.5))
    # set title, dots, label, and font sizes
    sns.scatterplot(x=x_col, y=y_col, data=df, hue='Cluster', palette='colorblind', ax=ax, s=10)
    ax.set_title("Customer Clusters", fontsize=8)
    ax.set_xlabel(x_col, fontsize=6)
    ax.set_ylabel(y_col, fontsize=6)
    ax.tick_params(axis='both', labelsize=5)
    ax.legend(title='Cluster', fontsize=5, title_fontsize=6)
    # display the plot in Streamlit
    st.pyplot(fig, use_container_width=False)

# Calculates Within-Cluster Sum of Squares (WCSS) for different values of k
def elbow_method(df, features, k_range):
    WCSS = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42).fit(df[features])
        WCSS.append(model.inertia_)
    return WCSS

# Calculates WCSS (Within-Cluster Sum of Squares) for different k values,
# creates a DataFrame, and displays it in Streamlit.
def calculate_wcss_and_display(df, features, k_range):
    # Calculate WCSS using elbow method
    wcss_scores = elbow_method(df, features, k_range)
    # Create WSS DataFrame
    wss = pd.DataFrame({
        'cluster': list(k_range),
        'WSS_Score': wcss_scores
    })
    # Display DataFrame in Streamlit
    st.subheader("WCSS (Within-Cluster Sum of Squares) for Different k Values")
    st.dataframe(wss, use_container_width=False) 
    return wss

# Calculates Silhouette Scores for different values of k
def silhouette_method(df, features, k_range):
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42).fit(df[features])
        labels = model.labels_
        score = silhouette_score(df[features], labels)
        scores.append(score)
    return scores

# Displays an Elbow Plot (WCSS vs number of clusters)
def plot_elbow(wss_df):
    # add subheader
    st.subheader("Elbow Plot")
    # create a figure
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(wss_df['cluster'], wss_df['WSS_Score'], marker='o', linestyle='-')
    # set fontsize
    ax.set_xlabel('No. of clusters', fontsize=8)
    ax.set_ylabel('WSS Score', fontsize=8)
    ax.set_title('Elbow Plot', fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    # display the plot in Streamlit without stretching
    st.pyplot(fig, use_container_width=False)

# Displays a Silhouette Plot (Silhouette Score vs number of clusters)
def plot_silhouette(df, cluster_col, score_col):
    # write subheader
    st.subheader("Silhouette Plot")
    # create a figure
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(df[cluster_col], df[score_col], marker='o', linestyle='-')
    # set fontsize
    ax.set_xlabel('No. of clusters', fontsize=8)
    ax.set_ylabel('Silhouette Score', fontsize=8)
    ax.set_title('Silhouette Plot', fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    # display the plot in Streamlit without stretching
    st.pyplot(fig, use_container_width=False)
    
# Calculates silhouette scores on selected features
def calculate_silhouette_for_features(df, features, k_range):
    K = []
    ss = []
    for i in k_range:
        kmodel = KMeans(n_clusters=i, random_state=42).fit(df[features])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[features], ypred)
        K.append(i)
        ss.append(sil_score)
    
    result_df = pd.DataFrame({
        'cluster': K,
        'Silhouette_Score': ss
    })
    return result_df

# plot silhouette scores for the selected features
def plot_silhouette_for_features(result_df):
    st.subheader("Plot the Silhouette Plot")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(result_df['cluster'], result_df['Silhouette_Score'], marker='o', linestyle='-')
    ax.set_xlabel('No. of clusters', fontsize=8)
    ax.set_ylabel('Silhouette Score', fontsize=8)
    ax.set_title('Silhouette Plot', fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    st.pyplot(fig, use_container_width=False)