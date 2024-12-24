# Student Performance Indicator

## Problem Statement
This project aims to understand how students' performance (test scores) is affected by various factors such as:
- Gender
- Ethnicity
- Parental level of education
- Lunch type
- Test preparation course

## Project Workflow
This project follows the typical lifecycle of a Machine Learning project:

1. **Understanding the Problem Statement**
2. **Data Collection**
3. **Data Checks**
4. **Exploratory Data Analysis (EDA)**
5. **Data Pre-Processing**
6. **Model Training**
7. **Model Selection and Evaluation**

---

## 1. Problem Statement
The goal of this project is to analyze the factors influencing students' performance in exams, with a focus on scores in mathematics, reading, and writing.

## 2. Data Collection
- **Dataset Source**: [Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Data Details**: The dataset consists of 8 columns and 1000 rows.

### 2.1 Import Data and Required Packages
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

### 2.2 Importing the Dataset
```python
df = pd.read_csv('data/raw.csv')
df.head()
```

### 2.3 Dataset Structure
The dataset contains the following columns:
- **gender**: sex of the students (Male/Female)
- **race_ethnicity**: ethnicity of the students (Group A, B, C, D, E)
- **parental_level_of_education**: highest education level of the parents (e.g., Bachelor's degree, Some College)
- **lunch**: type of lunch (Standard/Free/Reduced)
- **test_preparation_course**: whether the student completed a test preparation course (Completed/None)
- **math_score**: score in math
- **reading_score**: score in reading
- **writing_score**: score in writing

## 3. Data Checks to Perform
- **Missing Values**: Checked and confirmed there are no missing values.
- **Duplicates**: No duplicate rows found.
- **Data Types**: All columns have appropriate data types.
- **Unique Values**: Checked the number of unique values for categorical columns.
- **Statistics**: Statistical summary for numerical features (Math, Reading, Writing).

### 3.1 Check for Missing Values
```python
df.isna().sum()
```

### 3.2 Check for Duplicates
```python
df.duplicated().sum()
```

### 3.3 Data Types and Summary
```python
df.info()
```

### 3.4 Number of Unique Values
```python
df.nunique()
```

### 3.5 Statistical Summary
```python
df.describe()
```

## 4. Exploratory Data Analysis (EDA)

### 4.1 Explore Data

- Categorical variables: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`.
- Numerical variables: `math_score`, `reading_score`, `writing_score`.

```python
df.head(2)
```

### 4.2 Add New Columns for Analysis
- **Total Score**: Sum of math, reading, and writing scores.
- **Average Score**: Total score divided by 3.

```python
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score'] / 3
```

### 4.3 Performance Insights

- **Full Marks**: Count of students who scored full marks in each subject.
- **Low Marks**: Count of students who scored less than 20 in each subject.

### 4.4 Visualizing Data

#### 4.4.1 Histogram & Kernel Distribution Estimation (KDE)
Visualize the distribution of average scores and total scores using histograms and KDE plots.

```python
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='average', bins=30, kde=True, color='g')
plt.subplot(122)
sns.histplot(data=df, x='average', kde=True, hue='gender')
plt.show()
```

#### 4.4.2 Gender-Based Performance
```python
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='total score', bins=30, kde=True, color='g')
plt.subplot(122)
sns.histplot(data=df, x='total score', kde=True, hue='gender')
plt.show()
```

### Insights
- **Gender Insights**: Female students tend to perform better than male students.
- **Lunch Insights**: Students who have a standard lunch perform better than those with a free/reduced lunch.

#### 4.4.3 Parental Education Influence
```python
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.histplot(data=df, x='average', kde=True, hue='parental_level_of_education')
plt.subplot(142)
sns.histplot(data=df[df['gender'] == 'male'], x='average', kde=True, hue='parental_level_of_education')
plt.subplot(143)
sns.histplot(data=df[df['gender'] == 'female'], x='average', kde=True, hue='parental_level_of_education')
plt.show()
```

### Insights:
- **Parental Education**: Parental education does not strongly correlate with the student's performance, except for male students whose parents have an associate's or master's degree.

#### 4.4.4 Ethnicity-Based Performance
```python
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.histplot(data=df, x='average', kde=True, hue='race_ethnicity')
plt.subplot(142)
sns.histplot(data=df[df['gender'] == 'female'], x='average', kde=True, hue='race_ethnicity')
plt.subplot(143)
sns.histplot(data=df[df['gender'] == 'male'], x='average', kde=True, hue='race_ethnicity')
plt.show()
```

### Insights:
- **Ethnicity**: Students from Group A and Group B tend to perform poorly compared to others.

## Conclusion
- **Best Performance**: Students excel in reading, while math scores tend to be lower.
- **Gender Impact**: Females tend to outperform males.
- **Lunch Type**: Students who have a standard lunch tend to score better.

## Future Work
- **Model Training**: Implement machine learning models to predict student performance based on these features.
- **Data Pre-processing**: Feature scaling, encoding categorical variables, etc.
- **Model Evaluation**: Use various evaluation metrics like accuracy, precision, recall, etc.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` provides an overview of the dataset, problem statement, EDA process, insights, and visualizations. You can modify it further to add more details as per your project’s requirements.
