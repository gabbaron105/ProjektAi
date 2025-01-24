import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

X = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')

print("Sprawdzanie braków w danych...")
print(X.isnull().sum())

print("\nPrzykład danych:")
print(X.head())

# 3. Podział danych na zbiór treningowy i testowy
print("\nDzielenie danych na treningowe i testowe...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Zakodowanie zmiennych kategorycznych
print("\nKodowanie zmiennych kategorycznych...")
label_encoders = {}
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le

print("\nSkalowanie danych...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nRedukcja wymiarowości za pomocą PCA...")
pca = PCA(n_components=10)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nTworzenie i trenowanie modelu...")
model = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
model.fit(X_train_pca, y_train.values.ravel())

print("\nOcena modelu za pomocą walidacji krzyżowej...")
cv_scores = cross_val_score(model, X_train_pca, y_train.values.ravel(), cv=5, scoring='accuracy')
print(f"Średnia dokładność walidacji krzyżowej: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

print("\nPredykcja na zbiorze testowym...")
y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nDokładność na zbiorze testowym: {accuracy:.2f}")
print("\nMacierz pomyłek:")
print(conf_matrix)
print("\nRaport klasyfikacji:")
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Macierz pomyłek')
plt.xlabel('Przewidywane')
plt.ylabel('Rzeczywiste')
plt.show()
