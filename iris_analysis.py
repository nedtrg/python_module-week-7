# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Optional: Use seaborn styling
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully.\n")
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

except Exception as e:
    print("An error occurred while loading the dataset:", e)
    exit()

# Task 2: Basic Data Analysis
print("\nBasic statistics of numerical columns:")
print(df.describe())

print("\nMean of each feature grouped by species:")
group_means = df.groupby('species').mean()
print(group_means)

# Task 3: Data Visualization

# 1. Line chart (simulated trend over index as time isn't present)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart - Sepal Length over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(6, 4))
group_means['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title("Bar Chart - Avg Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='lightgreen', edgecolor='black')
plt.title("Histogram - Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot - Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2')
plt.title("Scatter Plot - Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Observations
print("\nObservations:")
print("- Versicolor and virginica tend to have longer petal lengths than setosa.")
print("- Sepal width appears to be normally distributed with slight skew.")
print("- Sepal length shows a strong relationship with petal length.")
