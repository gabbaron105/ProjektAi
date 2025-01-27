import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Loading data...")
df = pd.read_csv('heart.csv')

print("\nDataset overview:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nChecking for missing values...")
print(df.isnull().sum())

X = df.drop(columns=['target'])
y = df['target']               

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print("\nNumeric features:", numeric_features.tolist())
print("Categorical features:", categorical_features.tolist())

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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  
)

print("\nTraining the model...")
pipeline.fit(X_train, y_train)

print("\nEvaluating the model...")
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

X_train_preprocessed = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_preprocessed = pipeline.named_steps['preprocessor'].transform(X_test)

pd.DataFrame(X_train_preprocessed).to_csv('X_train_processed.csv', index=False)
pd.DataFrame(X_test_preprocessed).to_csv('X_test_processed.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=['target'])
pd.DataFrame(y_test).to_csv('y_test.csv', index=False, header=['target'])

print("\nProcessed data and labels saved to files:")
print(" - X_train_processed.csv")
print(" - X_test_processed.csv")
print(" - y_train.csv")
print(" - y_test.csv")
