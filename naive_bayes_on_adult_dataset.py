import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

#  Import dataset

df = pd.read_csv('../datafile/adult.csv')

 # Exploratory data analysis ¶

print(df.shape)
print(df.head())

#view summary of dataset
print(df.info())
# there is no null values in dataset

# Explore categorical variables:
    # find categorical variables

# categorical = [var for var in df.columns if df[var].dtype=='O']

# print('There are {} categorical variables\n'.format(len(categorical)))

# print('The categorical variables are :\n\n', categorical)
categorical = [var for var in df.columns if df[var].dtype=='O']
print('there are {} categorical variables\n'.format(len(categorical)))
print('the categorical variables are : \n\n', categorical)

print(df[categorical].head().to_string())

print('\n\n')
print(categorical)
print('\n\n')
# Explore problems within categorical variables:
    # First, I will explore the categorical variables.

    # Missing values in categorical variables
print(df[categorical].isnull().sum())

print('\n\n')

# Frequency counts of categorical variables
# Now, I will check the frequency counts of categorical variable
for var in categorical:

    print(df[var].value_counts())

# view frequency distribution of categorical variables
print('\n\n\n')
for var in categorical:

    print(df[var].value_counts()/np.float(len(df)))
    print(df[var].value_counts()/np.float(len(df)))

print(np.float(len(df)))
df['workclass'].replace('?', np.NaN, inplace=True)
# We can see that there are 1836 values encoded as ? in workclass variable. I will replace these ? with NaN.
# replace '?' values in workclass variable with `NaN`
print(df['workclass'].replace('?', np.NaN, inplace=True))
print(df.workclass.value_counts())

# Explore occupation variable
    # check labels in occupation variable
print(df['occupation'].unique())
print(df['occupation'].value_counts())
print(df['occupation'].replace('?',np.NaN, inplace=True))
print('\n\n')
print(df.occupation.value_counts())

print('\n\n')
#Explore native_country variable:
    # check labels in native_country variable

df['native-country'].replace('?', np.NaN, inplace=True)
print(df['native-country'].value_counts())

print('\n\n')
# Check missing values in categorical variables again¶
print(df[categorical].isnull().sum())

# Explore Numerical Variables:
        # find numerical variable

numerical = [var for var in df.columns if df[var].dtype!='O']
print('there are {} numerical values\n'.format(len(numerical)))

print('the numerical variables are=', numerical)
print(df[numerical].head())

# Missing values in numerical variables
print(df[numerical].isnull().sum())
# We can see that all the 6 numerical variables do not contain missing values.

# 8. Declare feature vector and target variable ¶
x = df.drop(['income'],axis=1)
y = df.income
print(y)

# 9. Split data into separate training and test set ¶
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# check the shape of X_train and X_test
print(x_train.shape)
print(x_test.shape)

# 10. Feature Engineering
#   check data types in X_train

print(x_train.dtypes)

# display categorical variables
print('\n\n')
# categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical = [var for var in x_train.columns if x_train[var].dtypes == 'O']
print(categorical)

# display numerical variables

numerical = [var for var in x_train.columns if x_train[var].dtypes!='O']
print(numerical)
print('\n\n')
# print percentage of missing values in the categorical variables in training set

# X_train[categorical].isnull().mean()
print(x_train[categorical].isnull().mean())

# print categorical variables with missing data
print('\n\n')
# for col in categorical:
#     if X_train[col].isnull().mean()>0:
#         print(col, (X_train[col].isnull().mean()))

for var in categorical:
    if x_train[var].isnull().mean()>0:
        print(var,(x_train[var].isnull().mean()))

# impute missing categorical variables with most frequent value
for df2 in [x_train,x_test]:
    df2['workclass'].fillna(x_train['workclass'].mode()[0],inplace=True)
    df2['occupation'].fillna(x_train['occupation'].mode()[0],inplace=True)
    df2['native-country'].fillna(x_train['native-country'].mode()[0],inplace=True)

# check missing values in categorical variables in X_train

print(x_train[categorical].isnull().sum())
print('\n\n')
# check missing values in categorical variables in X_test
print(x_test[categorical].isnull().sum())

# As a final check, I will check for missing values in X_train and X_test.

    # check missing values in X_train
print(x_train.isnull().sum())

print('\n\n')
print(x_test.isnull().sum())

# Encode categorical variables¶
# print categorical variables
# import category encoders

# # import category_encoders as ce
# # encode remaining variables with one-hot encoding

# # encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
# #                                  'race', 'sex', 'native_country'])

# # X_train = encoder.fit_transform(X_train)

# # X_test = encoder.transform(X_test)
# X_train.head()

import category_encoders as ce
#encode remaining variables with one hot encoding
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])
print(encoder)




















































































































































