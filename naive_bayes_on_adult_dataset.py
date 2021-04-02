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
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                 'race', 'gender', 'native-country'])
print(encoder)
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)
print(x_train.shape)
print(x_test.shape)

# 11. Feature Scaling
    # Table of Contents
cols = x_train.columns
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])
x_train.head()

# 12. Model training:
    # Table of Contents
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(x_train, y_train)
# 13. Predict the results

y_pred = gnb.predict(x_test)
print('the value of x_test',x_test)
print(y_pred)
print('\n\n')

# 14. Check accuracy score
# Table of Contents
from sklearn.metrics import accuracy_score
print('model accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))

# Compare the train-set and test-set accuracy
# Now, I will compare the train-set and test-set accuracy to check for overfitting.


print('\n\n')
y_pred_train = gnb.predict(x_train)
print(y_pred_train)
print('training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))


# Check for overfitting and underfitting
    # print the scores on training and test set

print('training set score: {:.4f}'.format(gnb.score(x_train,y_train)))
print('test set score:{:.4f}'.format(gnb.score(x_test,y_test)))

print('\n\n')
#Compare model accuracy with null accuracy¶
print(y_test.value_counts())


# check null accuracy score
null_accuracy = ( 11138/( 11138+3515))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
print('null accuracy score:')

print('\n\n')
# 15. Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

plt.show()

# 16. Classification metrices
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Classification accuracy
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN)/(TP+TN+FP+FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Classification error
# print classification error

classification_error = (FP+FN)/float(TP+TN+FP+FN)
print('Classification_error : {0:0.4f}'.format(classification_error))

#Precision
# print precision score

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
print('\n\n')

#Recall
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#True Positive Rate
#True Positive Rate is synonymous with Recall.

true_positive_rate = TP /(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


#false positive rate
false_positive_rate = FP/float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

# Specificity

specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

# f1-score
# f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.


# 17. Calculate class probabilities
# Table of Contents
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(x_test)[0:10]
print(y_pred_prob)
print('\n\n')
# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50k', 'prob of - >50k'])
print(y_pred_prob_df)

print('\n\n')
# print the first 10 predicted probabilities for class 1 - Probability of >50K

# gnb.predict_proba(X_test)[0:10, 1]
print(gnb.predict_proba(x_test)[0:10,1])

# store the predicted probabilities for class 1 - Probability of >50K
y_pred1 = gnb.predict_proba(x_test)[:, 1]
plt.hist(y_pred1)

# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.show()
# Observations
# We can see that the above histogram is highly positive skewed.
# The first column tell us that there are approximately 5700 observations with probability between 0.0 and 0.1 whose salary is <=50K.
# There are relatively small number of observations with probability > 0.5.
# So, these small number of observations predict that the salaries will be >50K.
# Majority of observations predcit that the salaries will be <=50K.

# 18. ROC - AUC ¶
from sklearn.metrics import  roc_curve
fpr,tpr,thresholds = roc_curve(y_test, y_pred1, pos_label='>50K')
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.
# ROC AUC
# ROC AUC stands for Receiver Operating Characteristic - Area Under Curve. It is a technique to compare classifier performance. In this technique, we measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
# So, ROC AUC is the percentage of the ROC plot that is underneath the curve.

# compute ROC AUC
from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test,y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

# Interpretation
# ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
# ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.

# calculate cross-validated ROC AUC
from sklearn.model_selection import  cross_val_score
Cross_validated_ROC_AUC = cross_val_score(gnb , x_train , y_train , cv=5 ,scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# 19. k-Fold Cross Validation
    # Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, x_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# We can summarize the cross-validation accuracy by calculating its mean.
    # compute Average cross-validation score
print('Average cross-validation score: {:.4f}'.format(scores.mean()))

# Interpretation
# Using the mean cross-validation, we can conclude that we expect the model to be around 80.63% accurate on average.

# If we look at all the 10 scores produced by the 10-fold cross-validation, we can also conclude that there is a relatively small variance in the accuracy between folds, ranging from 81.35% accuracy to 79.64% accuracy. So, we can conclude that the model is independent of the particular folds used for training.

# Our original model accuracy is 0.8083, but the mean cross-validation accuracy is 0.8063. So, the 10-fold cross-validation accuracy does not result in performance improvement for this model.

# 20. Results and conclusion
# Table of Contents

# In this project, I build a Gaussian Naïve Bayes Classifier model to predict whether a person makes over 50K a year. The model yields a very good performance as indicated by the model accuracy which was found to be 0.8083.
# The training-set accuracy score is 0.8067 while the test-set accuracy to be 0.8083. These two values are quite comparable. So, there is no sign of overfitting.
# I have compared the model accuracy score which is 0.8083 with null accuracy score which is 0.7582. So, we can conclude that our Gaussian Naïve Bayes classifier model is doing a very good job in predicting the class labels.
# ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a very good job in predicting whether a person makes over 50K a year.
# Using the mean cross-validation, we can conclude that we expect the model to be around 80.63% accurate on average.
# If we look at all the 10 scores produced by the 10-fold cross-validation, we can also conclude that there is a relatively small variance in the accuracy between folds, ranging from 81.35% accuracy to 79.64% accuracy. So, we can conclude that the model is independent of the particular folds used for training.
# Our original model accuracy is 0.8083, but the mean cross-validation accuracy is 0.8063. So, the 10-fold cross-validation accuracy does not result in performance improvement for this model.


































