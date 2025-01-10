import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
file_path = "./data.csv"

data = pd.read_csv(file_path)

data = data.dropna()

categorical_columns = ["Country", "Disease Name", "Disease Category", "Age Group", "Gender", "Treatment Type", "Availability of Vaccines/Treatment"]
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder() # sprbj oneHotEncoder
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

numerical_columns = ["Prevalence Rate (%)", "Incidence Rate (%)", "Mortality Rate (%)",
                     "Population Affected", "Healthcare Access (%)", "Doctors per 1000",
                     "Hospital Beds per 1000", "Average Treatment Cost (USD)",
                     "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
                     "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)"]

scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

output_file = "Prepared_dane.csv"
data.to_csv(output_file, index=False)

print(f"Przygotowane dane zostały zapisane w pliku {output_file}")
