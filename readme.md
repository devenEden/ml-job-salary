### MACHINE LEARNING 
This repository contains code for exploring and analyzing a salary dataset and implementing candidate models to predict monthly salaries. Below, you will find information about the dataset and the steps performed in the code.


### Group

| Name                 | Student Number | Registration Number |
|----------------------|----------------|---------------------|
| Feta Deven           | 2100706319     | 21/U/06319/EVE      |
| Okware Ayvan Manshur | 2100721691     | 21/U/21691/EVE      | 
| Omasuge Obadiah      | 2100719903     | 21/U/19903/EVE      |

### Dataset Description

The dataset used in this code represents information about individuals' salaries and relevant features. The columns in the dataset are as follows:

- **Age**: The age of the individual (numerical).
- **Gender**: The gender of the individual (categorical: Male, Female).
- **Education** Level: The education level of the individual (categorical: Bachelor's, Master's, PhD).
- **Job Title**: The job title of the individual (categorical).
- **Years of Experience**: The number of years of experience of the individual (numerical).
- **Salary**: The monthly salary of the individual (numerical).


#### Code Overview

The code performs the following steps:

1. Data Exploration:

    Reads the dataset from a CSV file.
    Displays the first few rows of the dataset.
    Retrieves the number of rows and columns in the dataset.
    Displays the column names.
    Shows the data types of each feature.
    Provides summary statistics for the numerical features.
    Displays unique values and their counts for categorical features.
    Checks for missing values in the dataset.

2. Data Visualization:

    Uses a helper function to visualize the relationship between the target variable (Salary) and the features "Years of Experience" and "Age".
    Candidate Model Implementation:

    Loads a sample dataset with preprocessed data for the purpose of demonstration.
    Separates the features (X) and the target variable (y).
    Encodes the categorical variables using LabelEncoder.
    Splits the data into training and testing sets.

    Implements the following candidate models using scikit-learn:
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    - Support Vector Machine Regression (SVR)


3. Model Evaluation:

    Makes predictions using the trained models on the testing data.
    Calculates the mean squared error (MSE) as the evaluation metric for each model.
    Prints the MSE values for each model.
    Determines the best performing algorithm based on the lowest MSE.
    Displays the name of the best performing algorithm and its corresponding MSE.
    Running the Code


### Install Dependecies
```shell
  pip install -r requirements.txt
```
Make sure to update the file path in the code to the location of your own dataset. The code assumes that the dataset is in a CSV file format.

### Usage
Run the main python file 
```shell
python src/main.py
```
Or Run the notebook

### Conclusion

This code provides a basic framework for exploring a salary dataset, implementing candidate models, and evaluating their performance. By analyzing the dataset and training different models, you can gain insights into the relationship between the features and the target variable (salary) and choose the best performing algorithm for predicting salaries.