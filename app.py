from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)

# Task 1
@app.route('/task1')
def task1():
    dataset = pd.read_csv(r"C:\Users\ACER\Downloads\internship\train - train.csv")

    X = dataset.drop(columns=["target"]) 
    y = dataset["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    def get_new_data_point():
        data_input = input("Enter the values for the new data point separated by commas: ")
        new_data_point = [float(value.strip()) for value in data_input.split(',')]
        return np.array(new_data_point).reshape(1, -1)

    new_data_point = get_new_data_point()

    new_data_point_scaled = scaler.transform(new_data_point)

    cluster_assigned = kmeans.predict(new_data_point_scaled)[0]
    cluster_center = kmeans.cluster_centers_[cluster_assigned]
    distance_to_center = np.linalg.norm(new_data_point_scaled - cluster_center)

    return render_template('tasks.html', task_num=1, cluster_assigned=cluster_assigned, distance_to_center=distance_to_center)


# Task 2
@app.route('/task2')
def task2():
    train_dataset = pd.read_csv(r"C:\Users\ACER\Downloads\internship\train - train.csv")
    test_dataset = pd.read_csv(r"C:\Users\ACER\Downloads\internship\test - test.csv")

    X_train = train_dataset.drop(columns=["target"])
    y_train = train_dataset["target"]
    X_test = test_dataset

    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train, y_train)
    y_pred_logistic = logistic_regression.predict(X_test)

    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)

    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)

    return render_template('tasks.html', task_num=2, y_pred_logistic=y_pred_logistic, y_pred_rf=y_pred_rf, y_pred_svm=y_pred_svm)


# Task 3
@app.route('/task3')
def task3():
    raw_data = pd.read_csv(r"C:\Users\ACER\Downloads\internship\rawdata - inputsheet.csv")

    datewise_duration = raw_data.groupby(['date', 'position']).agg({'time': 'sum'}).unstack().fillna(0)
    datewise_activities = raw_data.groupby(['date', 'activity']).size().unstack().fillna(0)
    output = pd.concat([datewise_duration, datewise_activities], axis=1)

    return render_template('tasks.html', task_num=3, output=output)


if __name__ == '__main__':
    app.run(debug=True)
