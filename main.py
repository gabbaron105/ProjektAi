import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

file_path = "./dane.csv"

# Wczytanie danych
data = pd.read_csv(file_path)

# Kolumny kategoryczne
categorical_columns = ["Country", "Disease Name", "Disease Category", "Age Group", "Gender", "Treatment Type", "Availability of Vaccines/Treatment"]

# Kolumny numeryczne
numerical_columns = ["Prevalence Rate (%)", "Incidence Rate (%)", "Mortality Rate (%)",
                     "Population Affected", "Healthcare Access (%)", "Doctors per 1000",
                     "Hospital Beds per 1000", "Average Treatment Cost (USD)",
                     "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
                     "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)"]

# Imputacja danych numerycznych (średnia)
numerical_imputer = SimpleImputer(strategy="mean")

# Imputacja danych kategorycznych (najczęstsza wartość)
categorical_imputer = SimpleImputer(strategy="most_frequent")

# Imputacja i przekształcanie danych
column_transformer = ColumnTransformer(
    transformers=[
        ('num_imputer', numerical_imputer, numerical_columns),  # Imputacja numeryczna
        ('cat_imputer', categorical_imputer, categorical_columns),  # Imputacja kategoryczna
        ('onehot', OneHotEncoder(drop='first'), categorical_columns),  # One-hot encoding
        ('scaler', StandardScaler(), numerical_columns)  # Skalowanie danych numerycznych
    ]
)

# Przekształcenie danych
data_transformed = column_transformer.fit_transform(data)

# Zapis przetworzonych danych do pliku
output_file = "Prepared_data.csv"
pd.DataFrame(data_transformed).to_csv(output_file, index=False)

print(f"Przygotowane dane zostały zapisane w pliku {output_file}")
