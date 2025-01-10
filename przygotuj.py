import pandas as pd
from sklearn.model_selection import train_test_split


file_path = "./Prepared_dane.csv"  
data = pd.read_csv(file_path)

X = data.drop(columns=["Disease Name"]) 
y = data["Disease Name"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Dane zostały podzielone i zapisane do plików: X_train.csv, X_test.csv, y_train.csv, y_test.csv.")
