import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Loading all the necessary packages
import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from app_module import functions as func

import warnings
warnings.filterwarnings("ignore")

try:
    # Streamlit page config
    st.set_page_config(page_title="Loan Eligibility App", layout="wide")
    logging.info("Page configuration set successfully.")
    
    # define color variables
    header1_color = "#ffb677"  # light orange
    header2_color = "#5f6caf"  # dark blue
    div_color = "#ffdfc4"  # extra light orange
    subheader_color = "#5f6caf"  # dark blue

    # set the title of the Streamlit app
    st.markdown(f"<h1 style='color: {header1_color};'>Project 3. Clustering Algorithms</h1>", unsafe_allow_html=True)

    # add subheader
    st.markdown(f"<h2 style='color: {subheader_color};'>Mall Customer Segmentation Model</h2>", unsafe_allow_html=True)
    logging.info("Page titles and headers rendered.")

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
    try:
        df = func.load_data('data/mall_customers.csv')
        logging.info(f"Data loaded successfully with shape {df.shape}.")
    except FileNotFoundError as e:
        logging.error("Data file not found.", exc_info=True)
        st.error("Error: Data file not found. Please check the path.")
        st.stop()
    except Exception as e:
        func.handle_error(e)
        st.stop()

    # # display the first five rows of the dataset in the app
    st.write('The dataset is loaded. The first five and last five records displayed below:')
    df_head_5 = df.head()
    df_tail_5 = df.tail()
    st.dataframe(df_head_5, use_container_width=False)
    st.dataframe(df_tail_5, use_container_width=False)

    # Displays the number of rows and columns
    func.display_data_shape(df, div_color)

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
    func.plot_pairplot(df, ['Age', 'Annual_Income', 'Spending_Score'], header1_color)

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
    # # Assign the cluster labels predicted by the k-means model (kmodel) to a new column called 'Cluster' in the DataFrame (df)
    
    try:
        kmodel = KMeans(n_clusters=5, random_state=42).fit(df[['Annual_Income', 'Spending_Score']])
        df['Cluster'] = kmodel.labels_
        logging.info("KMeans model trained successfully with 2 features.")
    except Exception as e:
        func.handle_error(e)
        st.stop()

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
    

    # check the dataset
    df_head_5 = df.head()
    st.dataframe(df_head_5, use_container_width=False)

    st.write("How many observations belong to each cluster")
    # check how many observations belong to each cluster
    df_cluster = df['Cluster'].value_counts()
    st.dataframe(df_cluster, use_container_width=False)

    # Cluster visualization
    func.plot_clusters(df, 'Annual_Income', 'Spending_Score')

    st.write("Visually we are able to see 5 clear clusters. Let's verify them using the Elbow and Silhouette Method")

    # add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>1. Elbow Method</h3>", unsafe_allow_html=True)
    st.markdown("""
                - We will analyze clusters from 3 to 8 and calculate the WCSS scores. The WCSS scores can be used to plot the Elbow Plot.

                - WCSS = Within Cluster Sum of Squares
                """)

    # Calculate WCSS and display results
    wss = func.calculate_wcss_and_display(df, ['Annual_Income', 'Spending_Score'], range(3, 9))

    # plot
    func.plot_elbow(wss)

    st.write("We get 5 clusters as a best value of k using the WSS method.")

    # add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Silhouette Measure</h3>", unsafe_allow_html=True)

    st.write("Check the value of K using the Silhouette Measure")

    # import silhouette_score
    from sklearn.metrics import silhouette_score

    # range of k values
    k_range = range(3, 9)

    # # calculate silhouette scores using your module function
    # ss = func.silhouette_method(df, ['Annual_Income', 'Spending_Score'], k_range)

    # # add the silhouette scores to the existing WSS DataFrame
    # wss['Silhouette_Score'] = ss

    # # display the updated DataFrame
    # st.dataframe(wss, use_container_width=False)
    
    try:
        ss = func.silhouette_method(df, ['Annual_Income', 'Spending_Score'], range(3, 9))
        wss['Silhouette_Score'] = ss
        st.dataframe(wss, use_container_width=False)
        func.plot_silhouette(wss, 'cluster', 'Silhouette_Score')
        logging.info("Silhouette scores calculated and plotted successfully.")
    except Exception as e:
        func.handle_error(e)
        st.stop()

    # add subheader
    st.markdown("""
                ##### Silhouette score is between -1 to +1
                """)

    st.write("closer to +1 means the clusters are better")

    # plot the silhouette scores
    func.plot_silhouette(wss, 'cluster', 'Silhouette_Score')

    st.write("Conclusion: Both Elbow and Silhouette methods gave the optimal value of k=5")

    # add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Now use all the available features and use the k-means model.</h3>", unsafe_allow_html=True)

    st.markdown("""
                - Remember, now you cannot visualise the clusters with more than 2 features.
                - So, the optimal number of clusters can be only determined by Elbow and Silhouette methods.
                """)

     # Store the number of clusters and their respective silhouette scores in a dataframe
    st.write("Store the number of clusters and their respective silhouette scores in a dataframe")
   
    st.write("Train a model on 'Age','Annual_Income','Spending_Score' features")
    st.write("Store the number of clusters and their respective silhouette scores in a dataframe")
    
    try:
        Variables3 = func.calculate_silhouette_for_features(df, ['Age', 'Annual_Income', 'Spending_Score'], range(3, 9))
        st.write("Silhouette scores for 3 features:")
        st.dataframe(Variables3, use_container_width=False)
        func.plot_silhouette_for_features(Variables3)
        logging.info("Silhouette scores for 3 features calculated successfully.")
    except Exception as e:
        func.handle_error(e)
        st.stop()

    # add subheader
    st.markdown("""
                ##### Conclusion: With 3 features we now have the optimal value of k=6
                """)
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")