import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers.getRelationship import getRelationship
# from sklearn.linear_model import LinearRegression

script_dir = os.path.dirname(__file__)  # Script directory
full_path = os.path.join(script_dir, '../dataset/Salary_Data.csv')

data = pd.read_csv(full_path)

# Display the first few rows of the dataset
print(data.head())

# Get the number of rows and columns in the dataset
print("Shape:", data.shape)

# Get the column names
print("Columns:", data.columns)

# Display the data types of each feature
print(data.dtypes)

# Get summary statistics of numerical features
print(data.describe())

# Get unique values and their counts for categorical features
for column in data.select_dtypes(include="object"):
    print(f"\n{column}:")
    print(data[column].value_counts())

# Check for missing values
print(data.isnull().sum())# Identify the target variable

target_variable = "Monthly Salary"
years_of_experience = "Years of Experience"

getRelationship(data=data,target=target_variable,dep=years_of_experience)


