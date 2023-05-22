import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers.getRelationship import getRelationship
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
age = "Age"
gender = "Gender"
educatation_level = "Education Level"
job_title = "Job Title"

getRelationship(data,target_variable,years_of_experience)
getRelationship(data,target_variable,age)
getRelationship(data,target_variable,educatation_level)
getRelationship(data,target_variable,job_title)
getRelationship(data,target_variable,gender)# Separate features (X) and target variable (y)
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
linear_regression_predictions = linear_regression.predict(X_test)
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
print("Linear Regression MSE:", linear_regression_mse)




