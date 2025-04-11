import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Loading all the necessary packages
import logging
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from app_module import functions as func
import warnings

warnings.filterwarnings("ignore")

try:
    
    #-------- page setting, header, intro ---------------------
    
    # Streamlit page config
    st.set_page_config(page_title="🛍️ Customer Clustering App", layout="centered")
    logging.info("Page configuration set successfully.")
    
    # define color variables
    header1_color = "#ffb677"  # light orange

    # set the title of the Streamlit app
    st.markdown(f"<h1 style='color: {header1_color};'>🛍️ Customer Clustering App</h1>", unsafe_allow_html=True)
    
    #-------- the app overview -----------------------------
    
    st.markdown("""
    ### Overview
    This **Customer Clustering App** helps you predict which **customer cluster group** you belong to based on your personal information.
    By using **K-Means Clustering**, the app groups customers into different segments based on key features like **Age**, **Annual Income**, and **Spending Score**.

    **Purpose of this app**:

    - Understand customer behavior patterns

    - Group customers with similar characteristics

    - Support marketing, targeting, and customer analysis activities

    Clusters are based on a pre-trained model using real mall customer data.
    """)
    
    #-------- user instructions -------------------------------

    st.markdown("""
    ### How to Use This App

    1. **Enter Your Details**
    Fill out the form with the following information:

        - **Age** (between 0 and 100)

        - **Annual Income** (in thousands of dollars)

        - **Spending Score** (on a scale of 1 to 100)

    2. **Submit**
    Click the "**Predict Cluster**" button to find out which customer group you belong to.

    3. **View Your Result**
    The app will immediately display your **Cluster Number** based on the trained model.

    4. **Interpret Your Cluster**
    Different clusters represent different customer profiles, which businesses can use for targeted strategies.
    """)
    
    #-------- the dataset loading -----------------------------

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
     
    #-------- select and scale features -----------------------------   
        
    # Select features for clustering
    features = ['Age', 'Annual_Income', 'Spending_Score'] 
    X = df[features]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    
     #-------- user input form ----------------------------- 

    st.markdown("---")
    st.subheader("🐦 Predict Your Cluster")

    with st.form(key="cluster_form"):
        st.write("Enter your details:")

        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Age", min_value=0, max_value=100, value=30)
        with col2:
            Annual_Income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)
        with col3:
            Spending_Score = st.slider("Spending Score (1-100)", 1, 100, 50)

        submit = st.form_submit_button("Predict Cluster")

    if submit:
        try:
            # Create DataFrame for input
            user_input = pd.DataFrame([[Age, Annual_Income, Spending_Score]], columns=features)

            # Scale input
            user_input_scaled = scaler.transform(user_input)

            # Predict cluster
            cluster_label = kmeans.predict(user_input_scaled)[0]
            
             #-------- success message & result ----------------------------- 

            # Display result
            st.success(f"✨ You belong to **Cluster {cluster_label}**!")
        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            st.error("Prediction failed. Please try again.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")