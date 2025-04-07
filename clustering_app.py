# Loading all the necessary packages
import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# define color variables
header1_color = "#ffb677"  # light orange
header2_color = "#5f6caf"  # dark blue
div_color = "#ffdfc4"  # extra light orange
subheader_color = "#5f6caf"  # dark blue

# set the title of the Streamlit app
st.markdown(f"<h1 style='color: {header1_color};'>Project 3. Clustering Algorithms</h1>", unsafe_allow_html=True)

# add subheader
st.markdown(f"<h2 style='color: {subheader_color};'>Mall Customer Segmentation Model</h2>", unsafe_allow_html=True)

# add the project scope and description
st.markdown(f"<h2 style='color: {header2_color};'>Project Scope</h2>", unsafe_allow_html=True)
st.markdown("""
            Malls are often indulged in the race to increase their customers and making sales. 
            To achieve this task machine learning is being applied by many malls already.
            It is amazing to realize the fact that how machine learning can aid in such ambitions. 
            The shopping malls make use of their customers’ data and develop ML models to target the right audience for right product marketing.

            **Your Role:**
            Mall Customer data is an interesting dataset that has hypothetical customer data. It puts you in the shoes of the owner of a supermarket. 
            You have customer data, and on this basis of the data, you have to divide the customers into various groups.

            **Goal:**
            Build an unsupervised clustering model to segment customers into correct groups.

            **Specifics:**
            - Machine Learning Task: Clustering model
            - Target Variable: N/A
            - Input Variables: Refer to the data dictionary below
            - Success Criteria: Cannot be validated beforehand
            """)

# add data dictionary
st.markdown(f"<h2 style='color: {header2_color};'>Data Dictionary:</h2>", unsafe_allow_html=True)
st.markdown("""
            - **CustomerID:** Unique ID assigned to the customer
            - **Gender:** Gender of the customer
            - **Age:** Age of the customer
            - **Income:** Annual Income of the customers in 1000 dollars
            - **Spending_Score:** Score assigned between 1-100 by the mall based on customer' spending behavior
            """)

# add header
st.markdown(f"<h2 style='color: {header2_color};'>Data Analysis and Data Prep</h2>", unsafe_allow_html=True)

# add subheader
st.markdown(f"<h3 style='color: {subheader_color};'>Reading the data</h3>", unsafe_allow_html=True)

# load the dataset from a CSV file located in the 'data' folder
df = pd.read_csv('data/mall_customers.csv')

# # display the first five rows of the dataset in the app
st.write('The dataset is loaded. The first five and last five records displayed below:')
st.write(df.head())
st.write(df.tail())

# create variables for rows and columns counts
rows_count = df.shape[0]
columns_count = df.shape[1]
# display dataset shape
st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
            The dataset contains:
            <ul>
             <li><strong>Rows:</strong> { rows_count }</li>
             <li><strong>Columns:</strong> { columns_count }</li>
            </ul>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Check some quick stats of the data
describe = df.describe()

# display subheader
st.write("Descriptive statistics for all columns:")
st.dataframe(describe)

# calculate the pairwise correlation between numerical columns in the DataFrame
# 'numeric_only=True' ensures that only numeric columns are considered for correlation
corr_matrix = df.corr(numeric_only=True)

# display subheader
st.write("Calculate the pairwise correlation between numerical columns in the DataFrame:")
# display the correlation matrix
st.dataframe(corr_matrix)

# Calculate the pairwise Spearman rank correlation between numerical columns in the DataFrame
# 'numeric_only=True' ensures only numeric columns are considered
# 'method="spearman"' uses Spearman correlation, which measures the monotonic relationship between variables
corr_matrix_spearman = df.corr(numeric_only=True, method='spearman')

# display subheader
st.write("Calculate the pairwise Spearman rank correlation between numerical columns in the DataFrame:")
# display the correlation matrix
st.dataframe(corr_matrix_spearman)

# let's plot a pairplot
# add a title before the plot
st.markdown(f"<h2 style='text-align: center; color: {header1_color};'>Pairplot of Age, Annual Income, and Spending Score</h2>", unsafe_allow_html=True)

# create the pairplot
pairplot_fig = sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])

# display the plot in Streamlit
st.pyplot(pairplot_fig)

st.markdown("""
            - As a mall owner you are interested in the customer spending score. 
            If you look at spending vs Age, you can observe that the spending score is high for customers between age 20-40, 
            and relatively low for customers beyond 40.
            - Remember, K-means clustering is sensitive to outliers. So, if you see any guilty outliers you should consider removing them.
            """)

# import kmeans model
from sklearn.cluster import KMeans

st.markdown("""
             We will build a model with only 2 features for now to visualize it, and later we will add more feature' and use the evaluation metric silhouette measure.
            """)

# display subheader
st.write("Cluster Centers for Customer Segments:")

# Let' train the model on spending_score and annual_income
kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])

# Retrieve the cluster centers
cluster_centers = kmodel.cluster_centers_

# Convert cluster centers to a DataFrame with meaningful column names
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['Annual_Income_Center', 'Spending_Score_Center'])

# Display the cluster centers
st.dataframe(cluster_centers_df)

# Check the cluster labels
kmodel_labels = kmodel.labels_

# # Display the labels
st.markdown(f"""         
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
    <h3 style='color: {subheader_color};'>Cluster Labels</h3> 
            <p>
                {", ".join(map(str, kmodel_labels))}
            </p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Put this data back in to the main dataframe corresponding to each observation
df['Cluster'] = kmodel.labels_

st.write("Put this data back in to the main dataframe corresponding to each observation")
