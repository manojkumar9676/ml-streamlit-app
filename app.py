import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Title
st.title("ML Verification Result Predictor")

# Load Data (cached)
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/static/public/713/data.csv"
    df = pd.read_csv(url)
    # Encode any categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    return df

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Prepare Data
X = df.drop("verification.result", axis=1)
y = df["verification.result"]

# Train models (cached)
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    return log_model, tree_model, X_test, y_test

log_model, tree_model, X_test, y_test = train_models(X, y)

# Model Selection
st.write("### Model Selection")
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])

# Accuracy Display
if model_choice == "Logistic Regression":
    y_pred = log_model.predict(X_test)
else:
    y_pred = tree_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {acc:.2f}")

# User Input Section
st.write("### Enter Input Data")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_df)
    else:
        prediction = tree_model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")