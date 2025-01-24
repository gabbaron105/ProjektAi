import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder

print("Loading training and test data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__ccp_alpha': [0.01, 0.05]
}

print("Starting Grid Search Cross Validation...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train.values.ravel())
y_test = le_y.transform(y_test.values.ravel())

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nCross Validation Scores:")
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

y_pred = grid_search.predict(X_test)

print("\nTest Set Accuracy:")
print(f"{accuracy_score(y_test, y_pred):.3f}")

target_names = le_y.inverse_transform(np.unique(y_train)).astype(str)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nSaving results...")
results = pd.DataFrame({
    'Actual': le_y.inverse_transform(y_test),
    'Predicted': le_y.inverse_transform(y_pred)
})
results.to_csv('results.csv', index=False)

if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
    feature_names = (numeric_features.tolist() + 
                    [f"{feature}_encoded" for feature in categorical_features])
    
    feature_importances = pd.DataFrame(
        grid_search.best_estimator_.named_steps['classifier'].feature_importances_,
        index=feature_names,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances)
    feature_importances.to_csv('feature_importances.csv')

print("Process completed.")