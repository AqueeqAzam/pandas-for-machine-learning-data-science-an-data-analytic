import pandas as pd
import numpy as np

# 📌 1. Loading and Inspecting Data
# Definition: Data wrangling starts with loading and inspecting raw data.
# Usage: Used in data analysis, preprocessing before machine learning.

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [25, np.nan, 35, 40, 28],  # Missing value in Age
    "City": ["New York", "Los Angeles", "Chicago", "Chicago", "New York"],
    "Salary": [50000, 60000, 70000, 80000, None]  # Missing salary
})
print("\n🌟 Initial Data:\n", df)

print("\n📋 Data Info:")
df.info()  # Check structure and missing values


# 📌 2. Handling Missing Values
# Definition: Missing values can be filled, dropped, or imputed.
# Usage: Used in data cleaning for predictive modeling.

# Count missing values
print("\n⚠️ Missing Values Count:\n", df.isnull().sum())

# Fill missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)  # Fill Age with mean
df["Salary"].fillna(df["Salary"].median(), inplace=True)  # Fill Salary with median
print("\n✅ After Filling Missing Values:\n", df)

# Drop rows with missing values (alternative)
df_cleaned = df.dropna()
print("\n🗑️ After Dropping Missing Values:\n", df_cleaned)


# 📌 3. Removing Duplicates
# Definition: Duplicate data skews analysis and should be handled.
# Usage: Used in cleaning customer records, financial transactions.

df_duplicate = df.append(df.iloc[1], ignore_index=True)  # Introduce a duplicate
print("\n🚨 Duplicate Rows:\n", df_duplicate[df_duplicate.duplicated()])

df_no_duplicates = df_duplicate.drop_duplicates()
print("\n✅ Data After Removing Duplicates:\n", df_no_duplicates)


# 📌 4. Transforming Data (Renaming Columns)
# Definition: Renaming makes column names more readable.
# Usage: Used in restructuring messy datasets.

df.rename(columns={"Name": "Employee Name", "Salary": "Annual Salary"}, inplace=True)
print("\n📝 Renamed Columns:\n", df)


# 📌 5. Changing Data Types
# Definition: Ensures the correct data types for computations.
# Usage: Used in optimizing memory usage in large datasets.

df["Age"] = df["Age"].astype(int)  # Convert float to integer
df["Salary"] = df["Salary"].astype(float)  # Ensure Salary is float
print("\n🔄 Updated Data Types:\n", df.dtypes)


# 📌 6. Filtering Data Based on Conditions
# Definition: Extracts data that meets specific criteria.
# Usage: Used in customer segmentation, fraud detection.

filtered_df = df[df["Age"] > 30]  # Employees older than 30
print("\n🔍 Filtered Employees (Age > 30):\n", filtered_df)


# 📌 7. Applying Functions to Columns
# Definition: Allows data transformation using functions.
# Usage: Used in feature engineering for machine learning.

df["Salary After Tax"] = df["Annual Salary"].apply(lambda x: x * 0.8)  # Deduct 20% tax
print("\n💰 Salary After Tax:\n", df)


# 📌 8. Handling Categorical Data (Encoding)
# Definition: Converts categorical text data into numerical format.
# Usage: Used in machine learning models.

df["City Code"] = df["City"].astype("category").cat.codes  # Encode categorical values
print("\n🏙️ Encoded City Column:\n", df)


# 📌 9. Binning Data into Categories
# Definition: Divides numerical data into bins or groups.
# Usage: Used in customer age segmentation.

bins = [0, 25, 35, 50]
labels = ["Young", "Adult", "Senior"]
df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
print("\n📊 Binned Age Groups:\n", df)


# 📌 10. Merging and Joining Datasets
# Definition: Combines multiple tables based on a common column.
# Usage: Used in database management and relational datasets.

extra_data = pd.DataFrame({
    "Employee Name": ["Alice", "Bob", "Charlie"],
    "Department": ["HR", "IT", "Finance"]
})
df_merged = pd.merge(df, extra_data, on="Employee Name", how="left")
print("\n🔗 Merged DataFrame:\n", df_merged)


# 📌 11. Reshaping Data (Pivot and Melt)
# Definition: Reshapes wide and long datasets.
# Usage: Used in time series analysis, financial modeling.

pivot_df = df.pivot_table(values="Annual Salary", index="Age Group", aggfunc="mean")
print("\n📊 Pivot Table (Average Salary by Age Group):\n", pivot_df)

melted_df = df.melt(id_vars=["Employee Name"], value_vars=["Age", "Annual Salary"])
print("\n🔄 Melted DataFrame:\n", melted_df)


# 📌 12. Grouping and Aggregating Data
# Definition: Groups data and applies aggregate functions.
# Usage: Used in sales reports, customer segmentation.

df_grouped = df.groupby("City")["Annual Salary"].mean()
print("\n🏙️ Average Salary by City:\n", df_grouped)


# 📌 13. Detecting and Replacing Outliers
# Definition: Identifies extreme values using the IQR method.
# Usage: Used in fraud detection, financial anomaly detection.

Q1 = df["Annual Salary"].quantile(0.25)
Q3 = df["Annual Salary"].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df["Annual Salary"] < (Q1 - 1.5 * IQR)) | (df["Annual Salary"] > (Q3 + 1.5 * IQR))

print("\n🚨 Outliers Detected:\n", df[outlier_condition])

# Replacing outliers with median salary
median_salary = df["Annual Salary"].median()
df.loc[outlier_condition, "Annual Salary"] = median_salary
print("\n✅ Data After Replacing Outliers:\n", df)


# 📌 14. Working with Date and Time Data
# Definition: Parses and processes date/time columns.
# Usage: Used in time series forecasting, financial analysis.

df["Join Date"] = pd.to_datetime(["2023-01-15", "2022-06-10", "2021-12-05", "2020-03-22", "2019-07-10"])
df["Year Joined"] = df["Join Date"].dt.year
print("\n📆 Processed Date Data:\n", df)


# 📌 15. Sampling Data
# Definition: Selects a subset of data randomly.
# Usage: Used in ML model training, survey data sampling.

sampled_df = df.sample(n=2)  # Randomly pick 2 rows
print("\n🎲 Random Sample:\n", sampled_df)


# 📌 16. Saving Cleaned Data
# Definition: Saves processed data for later use.
# Usage: Used in data pipelines, ML workflows.

df.to_csv("cleaned_data.csv", index=False)  # Save to CSV
df.to_pickle("cleaned_data.pkl")  # Save to Pickle format
print("\n💾 Data Saved Successfully!")
