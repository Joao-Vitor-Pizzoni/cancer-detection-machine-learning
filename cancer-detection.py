import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Dataset Loading
# X = features (biomarkers), y = labels (0: Malignant, 1: Benign)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Data Splitting
# Training 75% of the data, Testing 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Model Initialization
# Random Forest for robust classification
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Support Vector Machine (SVM) for linear/non-linear separation
svm_clf = svm.SVC(gamma=0.001)

# 4. Model Training
rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# 5. Prediction
rf_pred = rf_clf.predict(X_test)
svm_pred = svm_clf.predict(X_test)

# 6. Evaluation - SVM Performance
print(f"--- SVM Classification Report ---")
print(metrics.classification_report(y_test, svm_pred, target_names=data.target_names))

print(f"--- Random Florest Classification Report ---")
print(metrics.classification_report(y_test, rf_pred, target_names=data.target_names))

# 7. Error Metrics (Corrected comparison)
mse = mean_squared_error(y_test, svm_pred)
print(f"Model: {svm_clf.__class__.__name__}")
print(f"Mean Squared Error: {mse:.3f}\n")

mse_rf = mean_squared_error(y_test, svm_pred)
print(f"Model: {rf_clf.__class__.__name__}")
print(f"Mean Squared Error: {mse_rf:.3f}\n")

# 8. Confusion Matrix Visualization

disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, display_labels=data.target_names
)
disp.figure_.suptitle("Confusion Matrix: Random Forest")
plt.show()

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, svm_pred) # Comparação
disp.figure_.suptitle("Confusion Matrix: Support Vector Machine")
plt.show()


# 9. Data Exploration - Feature Distribution

plt.figure(figsize=(8, 6))
plt.hist(X[y == 0, 0], bins=30, alpha=0.5, label='Malignant', color='red')
plt.hist(X[y == 1, 0], bins=30, alpha=0.5, label='Benign', color='green')
plt.legend(loc='upper right')
plt.title('Feature Distribution: Mean Radius')
plt.xlabel('Radius Value')
plt.ylabel('Frequency')
plt.show()
