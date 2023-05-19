# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Module

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
%matplotlib inline


# Load the Data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

# Data Analysis
train_data.head()
train_data.describe()
train_data.info()

# Data Visualization

sns.countplot(train_data['Survived'])

sns.distplot(train_data['Age'])

class_fare = train_data.pivot_table(index="Pclass", values="Fare", aggfunc=np.sum)
class_fare.plot(kind="bar")
plt.xlabel("Pclass")
plt.ylabel("Total Fare")
plt.xticks(rotation=0)
plt.show()


data = len(train_data)
data

data2 = len(test_data)
data2

# Combine two dataframes
df = pd.concat([train_data, test_data], axis=0)
df = df.reset_index(drop=True)
df.head()
df.tail()

# find the null values
df.isnull().sum()

df = df.drop(columns=['Cabin'], axis=1)
df['Age'].mean()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

sns.heatmap(df.corr(), annot=True)
plt.show()

sns.barplot(data=train_data, x='Pclass', y='Fare', hue='Survived')

sns.barplot(data=train_data, x='Survived', y='Fare', hue='Pclass')

# drop unnecassary columns (Name, Ticket)
df = df.drop(columns=['Name','Ticket'], axis=1)
df.head()

# Label encoding
cols = ['Sex','Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()
# male = 1 , female=1, S=2, C=0

# Split Train Test
train_len = len(train_data)
train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]
train.head()
test.head()
X = train.drop(columns=['PassengerId', 'Survived'], axis=1)
Y = train['Survived']
X.head()

# Model Training
def classify(model,x,y):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy: ', model.score(x_test, y_test))
    score = cross_val_score(model,x,y,cv=7)
    print('CV Score: ', np.mean(score))
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model,X,Y)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model,X,Y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model,X,Y)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model,X,Y)

from xgboost import XGBClassifier
model = XGBClassifier()
classify(model,X,Y)

from lightgbm import LGBMClassifier
model = LGBMClassifier()
classify(model,X,Y)

from catboost import CatBoostClassifier
model = CatBoostClassifier(verbose=0)
classify(model,X,Y)

model = CatBoostClassifier()
model.fit(X,Y)

test.head()

X_test = test.drop(columns=['PassengerId', 'Survived'], axis=1)

X_test.head()

prediction = model.predict(X_test)
prediction
# Test Submission

sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sub.head()
sub['Survived'] = prediction
sub['Survived'] = sub['Survived'].astype('int')
sub.head()

sub.to_csv('submission.csv', index=False)

