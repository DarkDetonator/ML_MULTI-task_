 streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm imimportport SVC

# Function to perform KMeans clustering
def perform_kmeans(data_point):
    dataset = pd.read_csv("/home/username/Documents/train.csv")
    X = dataset.drop(columns=["target"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    new_data_point_scaled = scaler.transform([data_point])
    cluster_assigned = kmeans.predict(new_data_point_scaled)[0]
    cluster_center = kmeans.cluster_centers_[cluster_assigned]
    distance_to_center = np.linalg.norm(new_data_point_scaled - cluster_center)
    return f"The new data point belongs to cluster {cluster_assigned} because it is closest to the cluster center. Distance to cluster center: {distance_to_center}"

# Function to perform classification using logistic regression, random forest, and SVM
def perform_classification():
    train_dataset = pd.read_csv("/home/username/Documents/train.csv")
    test_dataset = pd.read_csv("/home/username/Documents/test.csv")
    X_train = train_dataset.drop(columns=["target"])
    y_train = train_dataset["target"]
    X_test = test_dataset
    logistic_regression = LogisticRegression(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    svm_classifier = SVC(random_state=42)
    logistic_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    svm_classifier.fit(X_train, y_train)
    y_pred_logistic = logistic_regression.predict(X_test)
    y_pred_rf = random_forest.predict(X_test)
    y_pred_svm = svm_classifier.predict(X_test)
    return {
        "Logistic Regression Predictions": y_pred_logistic,
        "Random Forest Predictions": y_pred_rf,
        "SVM Predictions": y_pred_svm
    }

# Function to perform data analysis
def perform_data_analysis():
    raw_data = pd.read_csv("/home/username/Documents/rawdata.csv")
    # Perform data analysis tasks and return the results
    # For example, calculate datewise total duration and number of activities

# Streamlit UI
st.title("Task Application")

task = st.selectbox("Select Task", ["KMeans Clustering", "Classification Models", "Data Analysis"])

if task == "KMeans Clustering":
    st.subheader("Perform KMeans Clustering")
    data_point_input = st.text_input("Enter the values for the new data point separated by commas:")
    if st.button("Perform Clustering"):
        data_point = [float(value.strip()) for value in data_point_input.split(',')]
        result = perform_kmeans(data_point)
        st.write(result)

elif task == "Classification Models":
    st.subheader("Perform Classification Models")
    if st.button("Perform Classification"):
        result = perform_classification()
        for model, predictions in result.items():
            st.write(f"{model} Predictions:")
            st.write(predictions)

elif task == "Data Analysis":
    st.subheader("Perform Data Analysis")
    if st.button("Perform Analysis"):
        result = perform_data_analysis()
        # Display the results of data analysis
        # For example, show datewise total duration and number of activities
