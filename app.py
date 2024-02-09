from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

app = Flask(__name__)

# Task 1: KMeans Clustering
@app.route("/task1", methods=["GET", "POST"])
def task1():
    result = None
    if request.method == "POST":
        dataset = pd.read_csv("/home/darkdeto/Downloads/train - train.csv")
        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_scaled)
        new_data_point = [float(value.strip()) for value in request.form['data_point'].split(',')]
        new_data_point_scaled = scaler.transform([new_data_point])
        cluster_assigned = kmeans.predict(new_data_point_scaled)[0]
        cluster_center = kmeans.cluster_centers_[cluster_assigned]
        distance_to_center = np.linalg.norm(new_data_point_scaled - cluster_center)
        result = f"The new data point belongs to cluster {cluster_assigned} because it is closest to the cluster center. Distance to cluster center: {distance_to_center}"
    return render_template("task1.html", task1_result=result)

# Task 2: Classification Models
@app.route("/task2", methods=["GET", "POST"])
def task2():
    result = None
    if request.method == "POST":
        train_dataset = pd.read_csv("/home/darkdeto/Downloads/train - train.csv")
        test_dataset = pd.read_csv("/home/darkdeto/Downloads/test - test.csv")
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
        pred_df_logistic = pd.DataFrame({"target": y_pred_logistic})
        pred_df_rf = pd.DataFrame({"target": y_pred_rf})
        pred_df_svm = pd.DataFrame({"target": y_pred_svm})
        pred_df_logistic.to_csv("logistic_regression_predictions.csv", index=False)
        pred_df_rf.to_csv("random_forest_predictions.csv", index=False)
        pred_df_svm.to_csv("svm_predictions.csv", index=False)
        result = "Predictions saved to CSV files."
    return render_template("task2.html", task2_result=result)

# Task 3: Data Analysis
@app.route("/task3")
def task3():
    raw_data = pd.read_csv("/home/darkdeto/Downloads/rawdata - inputsheet.csv")
    # Implement data analysis logic 
    #task 3

    datewise_duration = raw_data.groupby(['date', 'position']).agg({'time': 'sum'}).unstack().fillna(0)
    datewise_activities = raw_data.groupby(['date', 'activity']).size().unstack().fillna(0)
    output = pd.concat([datewise_duration, datewise_activities], axis=1)

    result = output
    return render_template("task3.html", task3_result=result)

if __name__ == "__main__":
    app.run(debug=True)