# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Step 2: Load the Dataset
# Replace the path with your dataset
data = pd.read_csv('titanic.csv')  # Change to your dataset file path

# Preview the dataset
print(data.head())

# Step 3: Handle Missing Data
# Check for missing values
print("Missing Data:\n", data.isnull().sum())

# Handle missing values:
# - For 'Age', fill missing values with the median (common for numerical data).
# - For 'Embarked', fill missing values with the mode (common for categorical data).
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Check again for missing values
print("Missing Data After Handling:\n", data.isnull().sum())

# Step 4: Exploratory Data Analysis (EDA)
# Generate summary statistics for numerical features
print(data.describe())

# Check data types to ensure correct format for each feature
print(data.dtypes)

# Step 5: Correlation Analysis
# Compute the correlation matrix between numerical variables
correlation_matrix = data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 6: Encoding Categorical Variables
# Label encoding for 'Sex' (binary column: Male = 0, Female = 1)
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# One-hot encoding for 'Embarked' (multiple categories: Categorical data)
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)  # drop_first=True removes multicollinearity

# Step 7: Scaling the Data
# Standardizing 'Age' and 'Fare' using StandardScaler (Z-score normalization)
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Alternatively, you can use Min-Max scaling if needed:
# min_max_scaler = MinMaxScaler()
# data[['Age', 'Fare']] = min_max_scaler.fit_transform(data[['Age', 'Fare']])

# Step 8: Outlier Detection (Using Boxplots)
# Plot boxplots to identify potential outliers in 'Age' and 'Fare'
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['Age', 'Fare']])
plt.title('Boxplot for Age and Fare')
plt.show()

# Detect outliers using IQR (Interquartile Range) method
Q1 = data[['Age', 'Fare']].quantile(0.25)
Q3 = data[['Age', 'Fare']].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers (values outside 1.5 * IQR range)
outliers = ((data[['Age', 'Fare']] < (Q1 - 1.5 * IQR)) | (data[['Age', 'Fare']] > (Q3 + 1.5 * IQR)))
print("Outliers detected:\n", outliers.sum())

# Step 9: Feature Selection (Optional)
# Here we remove irrelevant features such as 'Name' and 'Ticket' as they may not be useful for analysis
data.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Step 10: Visualization (Graphs)

# 1. Histogram for 'Age' (shows the distribution of age values)
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, color='blue', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Pairplot for multiple numerical features
sns.pairplot(data[['Age', 'Fare', 'Pclass', 'SibSp']], diag_kind='kde', height=2.5)
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()

# 3. Scatter plot for 'Age' vs 'Fare'
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Age'], y=data['Fare'], color='red')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# 4. Countplot for 'Survived' (To see the distribution of the target variable)
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.show()

# 5. Boxplot for 'Age' across different 'Pclass' (To explore how Age differs by Passenger Class)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Age', data=data)
plt.title('Age Distribution across Passenger Class')
plt.show()

# 6. Heatmap for correlation between numerical features (already shown above)

# Conclusion:
# At this stage, we have performed:
# - Missing data handling
# - Correlation analysis
# - Encoding of categorical variables
# - Scaling numerical variables
# - Various visualizations to explore distributions and relationships
# Now the data is ready for building machine learning models or further analysis.
