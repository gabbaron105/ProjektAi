import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


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

le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train.values.ravel())
y_test = le_y.transform(y_test.values.ravel())


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ensure target_names are strings
target_names = le_y.inverse_transform(np.arange(len(le_y.classes_))).astype(str)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=X_train.columns,
    columns=['Importance']
).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

results = pd.DataFrame({'Actual': le_y.inverse_transform(y_test),
                        'Predicted': le_y.inverse_transform(y_pred)})
results.to_csv('results.csv', index=False)
feature_importances.to_csv('feature_importances.csv')
