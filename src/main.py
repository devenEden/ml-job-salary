import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers.getRelationship import getRelationship
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

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

target_variable = "Salary"
years_of_experience = "Years of Experience"
age = "Age"
gender = "Gender"
educatation_level = "Education Level"
job_title = "Job Title"

getRelationship(data,target_variable,years_of_experience)
getRelationship(data,target_variable,age)

# Load the data
data = pd.DataFrame({
    'Age': [32.0, 28.0, 45.0, 36.0, 52.0],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Education Level': ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
    'Job Title': ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director'],
    'Years of Experience': [5.0, 3.0, 15.0, 7.0, 20.0],
    'Salary': [90000.0, 65000.0, 150000.0, 60000.0, 200000.0]
})

# Separate features (X) and target variable (y)
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Encode categorical variables
categorical_cols = ['Gender', 'Education Level', 'Job Title']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement candidate models
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

svm = SVR()
svm.fit(X_train, y_train)


# Evaluate the models
linear_regression_predictions = linear_regression.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)
random_forest_predictions = random_forest.predict(X_test)
svm_predictions = svm.predict(X_test)

linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
svm_predictions_mse =  mean_squared_error(y_test, svm_predictions)

print("Linear Regression MSE:",linear_regression_mse)
print("Decision Tree MSE:",decision_tree_mse )
print("Random Forest MSE:", random_forest_mse)
print("Support Vector Machine MSE:", random_forest_mse)

mse_values = [linear_regression_mse, decision_tree_mse, random_forest_mse, svm_predictions_mse]
algorithm_names = ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine",]

best_algorithm_index = mse_values.index(min(mse_values))
best_algorithm_name = algorithm_names[best_algorithm_index]
best_algorithm_mse = mse_values[best_algorithm_index]

print("Best Performing Algorithm:", best_algorithm_name)
print("MSE of Best Performing Algorithm:", best_algorithm_mse)


