import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Task 1 - K-Means Clustering
def task1():
    dataset = pd.read_csv("train - train.csv")

    X = dataset.drop(columns=["target"])  
    y = dataset["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # User input for a new data point
    data_input = input("Enter the values for the new data point separated by commas: ")
    new_data_point = np.array([float(value.strip()) for value in data_input.split(',')]).reshape(1, -1)

    new_data_point_scaled = scaler.transform(new_data_point)
    cluster_assigned = kmeans.predict(new_data_point_scaled)[0]
    
    # Scatter plot of clusters
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
    plt.scatter(new_data_point_scaled[:, 0], new_data_point_scaled[:, 1], color='red', marker='x', s=200, label="New Point")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='o', s=300, label="Cluster Centers")
    plt.title(f"K-Means Clustering (New Point Assigned to Cluster {cluster_assigned})")
    plt.legend()
    plt.show()

    print(f"New data point assigned to cluster: {cluster_assigned}")

# Task 2 - Classification Models
def task2():
    train_dataset = pd.read_csv("train - train.csv")
    test_dataset = pd.read_csv("test - test.csv")

    X_train = train_dataset.drop(columns=["target"])
    y_train = train_dataset["target"]
    X_test = test_dataset

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert categorical labels to numeric
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Train classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, solver='saga', random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42)
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_encoded)  # Fit models with scaled data
        preds = model.predict(X_test_scaled)
        predictions[name] = preds
        print(f"\n{name} Predictions (Encoded Labels):\n", preds)

    # Bar Chart of Prediction Counts
    plt.figure(figsize=(10, 5))
    for idx, (name, preds) in enumerate(predictions.items()):
        plt.subplot(1, 3, idx+1)
        plt.hist(preds, bins=np.arange(0, len(set(y_train_encoded))+2)-0.5, alpha=0.7, color=np.random.rand(3,))
        plt.title(name)
        plt.xlabel("Predicted Class (Encoded)")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

# Task 3 - Data Aggregation & Visualization
# Task 3 - Data Aggregation & Visualization
def task3():
    raw_data = pd.read_csv("rawdata - inputsheet.csv")

    # Convert 'date' column to datetime format
    raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce')

    # Convert 'time' column to numeric (fixes plotting issue)
    raw_data['time'] = pd.to_numeric(raw_data['time'], errors='coerce')

    # Drop any rows where date conversion failed
    raw_data = raw_data.dropna(subset=['date'])

    # Aggregation
    datewise_duration = raw_data.groupby(['date', 'position'])['time'].sum().unstack().fillna(0)
    datewise_activities = raw_data.groupby(['date', 'activity']).size().unstack().fillna(0)

    # Convert all aggregated values to numeric (ensures no string types)
    datewise_duration = datewise_duration.apply(pd.to_numeric, errors='coerce')

    # Combine data
    output = pd.concat([datewise_duration, datewise_activities], axis=1)

    # Debugging - Check Data Types
    print("\nChecking Data Types:\n", datewise_duration.dtypes)

    # Line Plot for Time Spent per Position
    plt.figure(figsize=(10, 5))

    for column in datewise_duration.columns:
        plt.plot(datewise_duration.index, datewise_duration[column], label=column)

    plt.title("Time Spent at Different Positions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Time Spent")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    print("\nAggregated Data:\n", output)



# Menu to Run Tasks
if __name__ == '__main__':
    while True:
        print("\nChoose a Task to Run:")
        print("1 - K-Means Clustering")
        print("2 - Classification Models")
        print("3 - Data Aggregation")
        print("4 - Exit")

        choice = input("Enter task number: ")
        if choice == "1":
            task1()
        elif choice == "2":
            task2()
        elif choice == "3":
            task3()
        elif choice == "4":
            print("Exiting program.")
            break
        else:
            print("Invalid choice! Please enter a number between 1 and 4.")
