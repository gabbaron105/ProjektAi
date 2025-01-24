import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Wczytanie danych
file_path ="data.csv"
data = pd.read_csv(file_path)

# Kolumny kategoryczne i numeryczne
categorical_columns = ["Country", "Disease Name", "Disease Category", "Age Group", "Gender", "Treatment Type", "Availability of Vaccines/Treatment"]
numerical_columns = ["Prevalence Rate (%)", "Incidence Rate (%)", "Mortality Rate (%)",
                     "Population Affected", "Healthcare Access (%)", "Doctors per 1000",
                     "Hospital Beds per 1000", "Average Treatment Cost (USD)",
                     "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
                     "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)"]

# Imputacja i przetwarzanie danych
numerical_imputer = SimpleImputer(strategy="mean")
categorical_imputer = SimpleImputer(strategy="most_frequent")

column_transformer = ColumnTransformer(
    transformers=[
        ('num_imputer', numerical_imputer, numerical_columns),
        ('cat_imputer', categorical_imputer, categorical_columns),
        ('onehot', OneHotEncoder(drop='first'), categorical_columns),
        ('scaler', StandardScaler(), numerical_columns)
    ]
)

# Transformacja danych
data_transformed = column_transformer.fit_transform(data)

# Debugowanie: Sprawdzenie kolumn wynikowych
encoded_feature_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(categorical_columns)
all_feature_names = numerical_columns + list(encoded_feature_names)
print(f"Expected feature count: {len(all_feature_names)}")
print(f"Transformed feature count: {data_transformed.shape[1]}")

# Debugowanie: Wyświetl szczegóły dla OneHotEncoder
print("OneHotEncoder feature names:")
print(encoded_feature_names)
print(f"Number of one-hot encoded columns: {len(encoded_feature_names)}")

# Debugowanie: Wyświetl podsumowanie przetworzonych danych
print(f"Total transformed data shape: {data_transformed.shape}")

# Naprawa: Dynamiczne ustalenie nazw kolumn
if data_transformed.shape[1] != len(all_feature_names):
    # Dopasuj dynamicznie liczbę kolumn
    all_feature_names = numerical_columns + list(range(len(numerical_columns), data_transformed.shape[1]))
    print("Warning: Adjusted feature names to match transformed data.")

# Stworzenie ramki danych
processed_data = pd.DataFrame(data_transformed, columns=all_feature_names)

# Kodowanie zmiennej docelowej
label_encoder = LabelEncoder()
processed_data["Disease Name"] = label_encoder.fit_transform(data["Disease Name"])

# Podział na cechy i etykiety
X = processed_data.drop(columns=["Disease Name"])
y = processed_data["Disease Name"]

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Zapis do CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Dane przetworzone i zapisane.")
