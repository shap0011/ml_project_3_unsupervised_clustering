import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.DEBUG,
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
def display_data_info(df, div_color):
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
    st.markdown(f"<h2 style='color: {title_color};'>Pairplot of {', '.join(cols)}</h2>", unsafe_allow_html=True)
    pairplot_fig = sns.pairplot(df[cols])
    pairplot_fig.fig.set_size_inches(4, 4)
    st.pyplot(pairplot_fig, use_container_width=False)

# Displays a scatter plot of clusters based on two features
def plot_clusters(df, x_col, y_col):
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue='Cluster', palette='colorblind', ax=ax, s=10)
    ax.set_title("Customer Clusters", fontsize=8)
    ax.set_xlabel(x_col, fontsize=6)
    ax.set_ylabel(y_col, fontsize=6)
    ax.tick_params(axis='both', labelsize=5)
    ax.legend(title='Cluster', fontsize=5, title_fontsize=6)
    st.pyplot(fig, use_container_width=False)

# Calculates Within-Cluster Sum of Squares (WCSS) for different values of k
def elbow_method(df, features, k_range):
    WCSS = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42).fit(df[features])
        WCSS.append(model.inertia_)
    return WCSS

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
    st.subheader("Elbow Plot")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(wss_df['cluster'], wss_df['WSS_Score'], marker='o', linestyle='-')
    ax.set_xlabel('No. of clusters', fontsize=8)
    ax.set_ylabel('WSS Score', fontsize=8)
    ax.set_title('Elbow Plot', fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    st.pyplot(fig, use_container_width=False)

# Displays a Silhouette Plot (Silhouette Score vs number of clusters)
def plot_silhouette(df, cluster_col, score_col):
    st.subheader("Silhouette Plot")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(df[cluster_col], df[score_col], marker='o', linestyle='-')
    ax.set_xlabel('No. of clusters', fontsize=8)
    ax.set_ylabel('Silhouette Score', fontsize=8)
    ax.set_title('Silhouette Plot', fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    st.pyplot(fig, use_container_width=False)