# README: Machine Learning Model Predictions Analysis

## Project Overview
This project analyzes the prediction distributions of three different machine learning models:
1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**

Each model's predictions are compared using histogram plots to visualize the frequency of predicted classes.

---

## Data Description
The aggregated dataset contains the following columns:
- **Inside**: Number of items detected inside a region.
- **inside**: Duplicate label (possibly a data inconsistency).
- **outside**: Number of items detected outside a region.
- **picked**: Number of items picked.
- **placed**: Number of items placed.

A sample of the data:
```plaintext
             Inside  inside  outside  picked  placed
date
2024-01-16     0.0     0.0      0.0      40      40
2024-01-17     0.0     0.0      0.0      10       9
2024-01-18     0.0     0.0      0.0      37      39
```

---

## Visualizations
![Figure_1](https://github.com/user-attachments/assets/cdb4d81d-c130-4e13-8f66-71062f8d6819)

Three histogram plots show the predicted class distributions for each model.
- **X-axis**: Predicted class (encoded labels).
- **Y-axis**: Frequency of predictions.

Observations:
- Some models predict specific classes more frequently.
- Logistic Regression and SVM have more balanced distributions.
- Random Forest exhibits strong peaks for certain classes, indicating a preference.

---

## Next Steps
- Investigate class imbalances.
- Overlay the true class distribution for comparison.
- Evaluate model accuracy and bias.

---

## Requirements
To reproduce the results, install the following dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

To run the script:
```bash
python analysis.py
```

---

## Contact
For questions or contributions, feel free to reach out!

