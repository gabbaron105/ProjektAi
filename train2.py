import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

print("Loading training and test data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

print("Preprocessing training data...")
def preprocess_data(X):
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)
    return X, label_encoders

X_train, label_encoders_train = preprocess_data(X_train)
X_test, _ = preprocess_data(X_test)

print("Encoding target variables...")
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train.values.ravel())
y_test = le_y.transform(y_test.values.ravel())

print("Initializing Decision Tree model...")
model = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1], 
    'random_state': [42]
}



print("Starting Grid Search...")
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Grid Search completed.")
best_model = grid_search.best_estimator_

print("Predicting on test data...")
y_pred = best_model.predict(X_test)

target_names = le_y.inverse_transform(np.arange(len(le_y.classes_))).astype(str)

print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importances = pd.DataFrame(
    best_model.feature_importances_,
    index=X_train.columns,
    columns=['Importance']
).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Zapis wyników
print("Saving results to CSV files...")
results = pd.DataFrame({'Actual': le_y.inverse_transform(y_test),
                        'Predicted': le_y.inverse_transform(y_pred)})
results.to_csv('results.csv', index=False)
feature_importances.to_csv('feature_importances.csv')
print("Results saved.")
