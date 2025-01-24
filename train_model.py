import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 1. Wczytanie danych
print("Loading training and test data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

print("\nTraining data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# 2. Rozdzielenie kolumn na numeryczne i kategoryczne
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

print("\nNumeric features:", numeric_features.tolist())
print("Categorical features:", categorical_features.tolist())

# 3. Tworzenie transformerów
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Enkodowanie etykiet
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train.values.ravel())
y_test = le_y.transform(y_test.values.ravel())

# 5. Tworzenie potoku z modelem RandomForest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Parametry do przeszukania
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

# 7. Grid Search
print("\nStarting Grid Search with Cross Validation...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1  
)

try:
    grid_search.fit(X_train, y_train)
    print("\nBest Parameters:")
    print(grid_search.best_params_)
except Exception as e:
    print("Error during GridSearchCV:", e)

# 8. Ocena modelu
try:
    y_pred = grid_search.best_estimator_.predict(X_test)
    print("\nTest Set Accuracy:")
    print(f"{accuracy_score(y_test, y_pred):.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_y.inverse_transform(np.unique(y_train)).astype(str)))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
except Exception as e:
    print("Error during model evaluation:", e)

# 9. Zapis wyników
try:
    results = pd.DataFrame({
        'Actual': le_y.inverse_transform(y_test),
        'Predicted': le_y.inverse_transform(y_pred)
    })
    results.to_csv('results.csv', index=False)
    print("\nResults saved to results.csv")
except Exception as e:
    print("Error during saving results:", e)

print("\nProcess completed.")
