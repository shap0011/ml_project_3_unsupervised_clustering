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

# Set page configuration
st.set_page_config(
    page_title="Loan Eligibility App",  # Your custom title
    layout="wide"                   # Or "wide" if you want more space
)

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
df_head_5 = df.head()
df_tail_5 = df.tail()
st.dataframe(df_head_5, use_container_width=False)
st.dataframe(df_tail_5, use_container_width=False)

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
st.dataframe(describe, use_container_width=False)

# calculate the pairwise correlation between numerical columns in the DataFrame
# 'numeric_only=True' ensures that only numeric columns are considered for correlation
corr_matrix = df.corr(numeric_only=True)

# display subheader
st.write("Calculate the pairwise correlation between numerical columns in the DataFrame:")
# display the correlation matrix
st.dataframe(corr_matrix, use_container_width=False)

# Calculate the pairwise Spearman rank correlation between numerical columns in the DataFrame
# 'numeric_only=True' ensures only numeric columns are considered
# 'method="spearman"' uses Spearman correlation, which measures the monotonic relationship between variables
corr_matrix_spearman = df.corr(numeric_only=True, method='spearman')

# display subheader
st.write("Calculate the pairwise Spearman rank correlation between numerical columns in the DataFrame:")
# display the correlation matrix
st.dataframe(corr_matrix_spearman, use_container_width=False)

# let's plot a pairplot
# add a title before the plot
st.markdown(f"<h2 style='color: {header1_color};'>Pairplot of Age, Annual Income, and Spending Score</h2>", unsafe_allow_html=True)

# Create the pairplot
pairplot_fig = sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])

# ✨ Resize the pairplot to make it smaller
pairplot_fig.fig.set_size_inches(4, 4)  # width, height in inches (adjust as needed)

# Display the plot in Streamlit
st.pyplot(pairplot_fig, use_container_width=False)

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
st.dataframe(cluster_centers_df, use_container_width=False)

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
st.write("Put this data back in to the main dataframe corresponding to each observation")
# Assign the cluster labels predicted by the k-means model (kmodel) to a new column called 'Cluster' in the DataFrame (df)
df['Cluster'] = kmodel.labels_

# check the dataset
df_head_5 = df.head()
st.dataframe(df_head_5, use_container_width=False)

st.write("How many observations belong to each cluster")
# check how many observations belong to each cluster
df_cluster = df['Cluster'].value_counts()
st.dataframe(df_cluster, use_container_width=False)

# Let' visualize these clusters
st.subheader("Cluster Visualization")

# Create a smaller figure
fig, ax = plt.subplots(figsize=(4, 2.5))

# Make dots smaller, reduce font sizes
sns.scatterplot(
    x='Annual_Income',
    y='Spending_Score',
    data=df,
    hue='Cluster',
    palette='colorblind',
    ax=ax,
    s=10  # <<< 's' is dot size (smaller number = smaller dots)
)

# Adjust labels and title font sizes
ax.set_title("Customer Clusters", fontsize=8)
ax.set_xlabel("Annual Income", fontsize=6)
ax.set_ylabel("Spending Score", fontsize=6)

ax.tick_params(axis='both', labelsize=5)

# Shrink the legend font size
ax.legend(title='Cluster', fontsize=5, title_fontsize=6)

# Display the plot in Streamlit
st.pyplot(fig, use_container_width=False)

st.write("Visually we are able to see 5 clear clusters. Let's verify them using the Elbow and Silhouette Method")

# add subheader
st.markdown(f"<h3 style='color: {subheader_color};'>1. Elbow Method</h3>", unsafe_allow_html=True)
st.markdown("""
            - We will analyze clusters from 3 to 8 and calculate the WCSS scores. The WCSS scores can be used to plot the Elbow Plot.

            - WCSS = Within Cluster Sum of Squares
            """)

# try using a for loop
st.subheader("WCSS (Within-Cluster Sum of Squares) for Different k Values")

# Initialize empty lists
K = []
WCSS = []

# Range of k values
k_range = range(3, 9)

# Calculate WCSS for each k
for k in k_range:
    kmodel = KMeans(n_clusters=k, random_state=42).fit(df[['Annual_Income', 'Spending_Score']])
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    K.append(k)

# Display results
wss = pd.DataFrame({
    'cluster': K,
    'WSS_Score': WCSS
})

st.dataframe(wss, use_container_width=False)

# Plot WCSS vs k
st.subheader("Elbow Plot")

# Create a figure
fig, ax = plt.subplots(figsize=(4, 2.5))


ax.plot(wss['cluster'], wss['WSS_Score'], marker='o', linestyle='-')

# labels, title, and ticks
ax.set_xlabel('No. of clusters', fontsize=8)
ax.set_ylabel('WSS Score', fontsize=8)
ax.set_title('Elbow Plot', fontsize=10)

# axis numbers
ax.tick_params(axis='both', labelsize=6)

# Display the plot in Streamlit without stretching
st.pyplot(fig, use_container_width=False)

st.write("We get 5 clusters as a best value of k using the WSS method.")

# add subheader
st.markdown(f"<h3 style='color: {subheader_color};'>Silhouette Measure</h3>", unsafe_allow_html=True)

st.write("Check the value of K using the Silhouette Measure")

# import silhouette_score
from sklearn.metrics import silhouette_score

# same as above, calculate sihouetter score for each cluster using a for loop

# try using a for loop
k = range(3,9) # to loop from 3 to 8
K = []         # to store the values of k
ss = []        # to store respective silhouetter scores
for i in k:
    kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
    ypred = kmodel.labels_
    sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
    K.append(i)
    ss.append(sil_score)
    
st.dataframe(ss, use_container_width=False)

# Store the number of clusters and their respective silhouette scores in a dataframe
st.write("Store the number of clusters and their respective silhouette scores in a dataframe")
wss['Silhouette_Score']=ss

st.dataframe(wss, use_container_width=False)

# add subheader
st.markdown("""
            ##### Silhouette score is between -1 to +1
            """)

st.write("closer to +1 means the clusters are better")

st.subheader("Silhouette Plot")

# Create a figure
fig, ax = plt.subplots(figsize=(4, 2.5))

# Plot Silhouette Score vs number of clusters
ax.plot(wss['cluster'], wss['Silhouette_Score'], marker='o', linestyle='-')

# Labels, title, and ticks
ax.set_xlabel('No. of clusters', fontsize=8)
ax.set_ylabel('Silhouette Score', fontsize=8)
ax.set_title('Silhouette Plot', fontsize=10)

# Axis numbers
ax.tick_params(axis='both', labelsize=6)

# Display the plot in Streamlit without stretching
st.pyplot(fig, use_container_width=False)

st.write("Conclusion: Both Elbow and Silhouette methods gave the optimal value of k=5")

# add subheader
st.markdown(f"<h3 style='color: {subheader_color};'>Now use all the available features and use the k-means model.</h3>", unsafe_allow_html=True)

st.markdown("""
            - Remember, now you cannot visualise the clusters with more than 2 features.
            - So, the optimal number of clusters can be only determined by Elbow and Silhouette methods.
            """)

# Train a model on 'Age','Annual_Income','Spending_Score' features
k = range(3,9)
K = []
ss = []
for i in k:
    kmodel = KMeans(n_clusters=i).fit(df[['Age','Annual_Income','Spending_Score']], )
    ypred = kmodel.labels_
    sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
    K.append(i)
    ss.append(sil_score)
    
# Store the number of clusters and their respective silhouette scores in a dataframe
st.write("Store the number of clusters and their respective silhouette scores in a dataframe")
Variables3 = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
st.dataframe(Variables3, use_container_width=False)

st.subheader("Plot the Silhouette Plot")

# Create a smaller figure
fig, ax = plt.subplots(figsize=(4, 2.5))

# Plot Silhouette Score vs number of clusters
ax.plot(Variables3['cluster'], Variables3['Silhouette_Score'], marker='o', linestyle='-')

# Smaller labels, title, and ticks
ax.set_xlabel('No. of clusters', fontsize=8)
ax.set_ylabel('Silhouette Score', fontsize=8)
ax.set_title('Silhouette Plot', fontsize=10)

ax.tick_params(axis='both', labelsize=6)

# Display the plot in Streamlit without stretching
st.pyplot(fig, use_container_width=False)

# add subheader
st.markdown("""
            ##### Conclusion: With 3 features we now have the optimal value of k=6
            """)
