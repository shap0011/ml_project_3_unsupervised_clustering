# Mall Customer Segmentation - Customer Clustering App 

This project builds an **Unsupervised Machine Learning model** to segment customers of a mall based on their spending habits, income, and age.

The project is deployed as an interactive **Streamlit web app** where you can explore the data, and validate the best number of clusters.

## Technologies Used

- [Streamlit](https://streamlit.io/) - For building the interactive web app
- [Pandas](https://pandas.pydata.org/) - For data manipulation
- [Logging](https://docs.python.org/3/library/logging.html) - For backend log management
- [Os](https://docs.python.org/3/library/os.html) - For setting STREAMLIT_LOG_LEVEL
- [Warnings](https://docs.python.org/3/library/warnings.html) - To ignore warnings

## Project Structure

- **.streamlit/**
  - `config.toml` — Theme setting
- `clustering_app.py` — Main Streamlit app
- **app_module/**
  - `__init__.py`
  - `functions.py` — All helper functions
- **data/**
  - `mall_customers.csv` — Raw dataset
- `requirements.txt` — List of Python dependencies
- `README.md` — Project documentation

## Dataset
The dataset contains the following features:

| Feature	      | Description                                  |
|-----------------|----------------------------------------------|
| CustomerID	  | Unique ID assigned to each customer          |
| Gender	      | Gender of the customer                       |
| Age	          | Age of the customer                          |
| Income	      | Annual Income (in $1000)                     |
| Spending_Score  | Spending score assigned by the mall (1-100)  |

## App Features
- **Data Loading:**

    - Load mall customer dataset

- **Clustering:**

    - Apply **K-Means Clustering** on:

        - 3 features: Age, Annual Income, and Spending Score


- **Cluster Validation:**

    - **Elbow Method:** Determine optimal number of clusters (k) based on WCSS (Within Cluster Sum of Squares)

    - **Silhouette Analysis:** Confirm the best value of k based on silhouette scores

- **Error Handling:**

    - Gracefully handle errors like missing data files or model failures

    - Log all errors and important information for debugging

## Key Learning Outcomes

- Practical application of **unsupervised learning** (clustering) with real-world data.

- Understanding how to **validate clustering models** using Elbow and Silhouette methods.

- Hands-on experience with **interactive data visualization** in Streamlit.

- Implementing **error handling** and **logging** in a professional-grade app.

## How to Run the App Locally

1. **Clone the repository**

```bash```
git clone https://github.com/shap0011/ml_project_3_unsupervised_clustering.git
cd ml_project_3_unsupervised_clustering

2. **Install the required packages**

```bash```
    pip install -r requirements.txt

3. **Run the App**

```bash```
streamlit run clustering_app.py

4. Open the URL shown (usually http://localhost:8501) to view the app in your browser!

## Deployment
The app is also deployed on Streamlit Cloud.
Click [![Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-clustering-app-shap0011.streamlit.app/) to view the live app.

## Author
Name: Olga Durham

LinkedIn: [\[Olga Durham LinkedIn Link\]](https://www.linkedin.com/in/olga-durham/)

GitHub: [\[Olga Durham GitHub Link\]](https://github.com/shap0011)


## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://miniature-space-yodel-r679jjj44jw3pq44.github.dev/)

## License

This project is licensed under the MIT License.  
Feel free to use, modify, and share it.  
See the [LICENSE](./LICENSE) file for details.