import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=30)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [1000],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, y_train.values.ravel())

print("Najlepsze parametry:", grid_search.best_params_)
model = grid_search.best_estimator_

y_pred = model.predict(X_test_pca)
print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))
