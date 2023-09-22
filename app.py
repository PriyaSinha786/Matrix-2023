import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
# Load client, services, and feedback data
clients_df = pd.read_csv("client_data.csv")
services_df = pd.read_csv("services_data.csv")
feedback_df = pd.read_csv("feedback_data.csv")

# Define a custom color palette for Risk Tolerance
risk_tolerance_colors = {
    "Low": "red",
    "Medium": "orange",
    "High": "green"
}

# Sidebar for selecting a specific client
st.sidebar.header("Select a Client")
selected_client = st.sidebar.selectbox("Choose a client:", clients_df["Client"])

# st.header("Customer Value Prediction")

# Customer value prediction for the selected client with dummy random data
# st.sidebar.header("Customer Value Prediction for Selected Client")

X = feedback_df[["Age", "Credit Score"]]
y = feedback_df["Satisfaction"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a dictionary to store regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor()  # Dummy model for demonstration
}

# Generate random data for demonstration
random_age = np.random.randint(0, 30)
random_credit_score = np.random.randint(300, 850)
predicted_satisfaction = round(np.random.uniform(1.0, 5.0), 1)  # Random satisfaction score


# Dummy data for model metrics (replace with actual model evaluation)
selected_model = "Lasso Regression"
dummy_metrics = {
    "Model": selected_model,
    "Mean Squared Error": np.random.rand(),
    "R-squared": np.random.rand(),
}

st.sidebar.header("Select a Regression Model")
selected_model = st.sidebar.selectbox("Choose a model:", ["Default","Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

st.sidebar.write("Customer Satisfaction:", predicted_satisfaction)

# # Display model metrics
# st.subheader("Model Metrics")
# st.dataframe(pd.DataFrame([dummy_metrics]))

# # Dummy data for customer value predictions (replace with actual predictions)
# dummy_predictions = {
#     "ClientID": ["Client1", "Client2", "Client3"],
#     "Predicted Satisfaction": np.random.rand(3),
# }

st.sidebar.markdown("---")
# Sidebar for new client inputs
st.sidebar.header("New Client Onboarding")

# Collect inputs from the new client
new_risk_tolerance = st.sidebar.selectbox("Risk Tolerance:", clients_df["Risk Tolerance"].unique())
new_financial_goals = st.sidebar.selectbox("Financial Goals:", clients_df["Financial Goals"].unique())
investment_horizon = st.sidebar.selectbox("Investment Horizon:", clients_df["Investment Horizon"].unique())
# new_income_level = st.sidebar.selectbox("Income Level:", clients_df["Income Level"].unique())
# new_age = st.sidebar.slider("Time Period:", 1, 80, 10)
investment_experience = st.sidebar.selectbox("Investment Experience:",clients_df["Investment Experience"].unique())
# new_credit_score = st.sidebar.slider("Credit Score:", 300, 850, 600)

# Display the recommended services for the new client when the user clicks on the recommendation section
show_recommendations = st.sidebar.checkbox("Show Recommended Services")

if show_recommendations:
    # Implement your recommendation algorithm here to suggest 1 or 2 services to the new client based on their inputs
    # For demonstration purposes, we'll randomly select 1 or 2 services from all available services.
    all_services = services_df
    
    # Randomly select 1 or 2 services
    num_services_to_recommend = np.random.choice([1, 2], p=[0.7, 0.3])  # Adjust probabilities as needed
    recommended_services = all_services.sample(n=num_services_to_recommend)
    
    # Display the recommended services in a table
    st.sidebar.header("Recommended Services for New Client")
    st.sidebar.dataframe(recommended_services[["Service", "Interest Rate", "Liquidity", "Fees"]])

# Create a page title
st.title("Client-Service Relationship Visualization")

# Show selected client data
st.header("Selected Client Data")
st.dataframe(clients_df[clients_df["Client"] == selected_client])

# Create graphs and numbers for all clients
st.header("All Clients View")
st.dataframe(clients_df)

# Create a bar chart showing the distribution of Risk Tolerance among all clients with the custom color palette
risk_tolerances = clients_df["Risk Tolerance"].value_counts()
fig1 = px.bar(risk_tolerances, x=risk_tolerances.index, y=risk_tolerances.values, labels={"x": "Risk Tolerance", "y": "Count"},
             title="Risk Tolerance Distribution", color=risk_tolerances.index, color_discrete_map=risk_tolerance_colors)
st.plotly_chart(fig1)

# Create a pie chart showing the distribution of Preferred Services among all clients with different colors
preferred_services = clients_df["Preferred Services"].str.split(", ", expand=True).stack().value_counts()
fig2 = px.pie(preferred_services, names=preferred_services.index, values=preferred_services.values,
              title="Preferred Services Distribution")
st.plotly_chart(fig2)

# Create a scatter plot showing the relationship between Age and Income Level with different colors
fig3 = px.scatter(clients_df, x="Time Period", y="ROI", color="Risk Tolerance",
                  labels={"Time Period": "Time Period", "ROI": "ROI"},
                  title="Time Period vs. Income Level (Colored by Risk Tolerance)")
st.plotly_chart(fig3)

# Create a histogram showing the distribution of Credit Scores with different colors
# fig4 = px.histogram(clients_df, x="Credit Score", color="Risk Tolerance",
#                     labels={"Credit Score": "Credit Score", "count": "Count"},
#                     title="Credit Score Distribution (Colored by Risk Tolerance)")
# st.plotly_chart(fig4)

# Regression models and customer value prediction
# st.header("Customer Value Prediction")

# Sidebar for selecting a regression model
# Sidebar for selecting a regression model
# st.sidebar.header("Select a Regression Model")
# selected_model = st.sidebar.selectbox("Choose a model:", ["Default","Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

# Prepare data for regression (use dummy data for demonstration)


# Select the chosen model
#selected_regression_model = models[selected_model]



# # Display customer value predictions
# st.subheader("Customer Value Predictions")
# st.dataframe(pd.DataFrame(dummy_predictions))

# # Dummy data for metric graphs (replace with actual data)
# metric_data = {
#     "Model": ["Model 1", "Model 2", "Model 3"],
#     "Mean Squared Error": [0.2, 0.3, 0.25],
#     "R-squared": [0.85, 0.78, 0.82],
# }

# # Create a bar chart for Mean Squared Error
# fig_metric1 = px.bar(metric_data, x="Model", y="Mean Squared Error", title="Mean Squared Error")
# st.plotly_chart(fig_metric1)

# # Create a bar chart for R-squared
# fig_metric2 = px.bar(metric_data, x="Model", y="R-squared", title="R-squared")
# st.plotly_chart(fig_metric2)

