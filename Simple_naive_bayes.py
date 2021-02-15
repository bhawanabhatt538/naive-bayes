

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


f1 = pd.read_json('../datafile/Sarcasm_Headlines_Dataset.json' , lines=True)
print(f1)

print('\n\n')
f2 = pd.read_json('../datafile/Sarcasm_Headlines_Dataset_v2.json' , lines=True)
print(f2)

print('\n\n')

df = pd.concat([f1,f2],axis=0,sort=False).drop('article_link',axis=1)
print(df)

print('\n\n')
print(df.info())

print(df.describe())

sns.histplot(data=df , x='is_sarcastic')
plt.show()

x = df['headline']
y = df['is_sarcastic']

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
print('\n\n')
X=cv.fit_transform(x)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
# print(X_train)
# print('\\n\n')
# print(X_test)
# print('\n\n')
# print(y_train)
# print('\n\n')
# print(y_test)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)

pre = nb.predict(X_test)
print(pre)

print('\n')
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pre))
print('\n\n')
print(classification_report(y_test,pre))
