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
import warnings

warnings.filterwarnings("ignore")


# Loads a CSV file into a pandas DataFrame
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def handle_error(e):
    logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    st.error("An unexpected error occurred. Please try again later.")
