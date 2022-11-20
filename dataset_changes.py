import pandas as pd
import numpy as np


url = 'https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv'
df = pd.read_csv(url,index_col=0)
print(df.head)
df['Embarked'] = df['Embarked'].replace(np.nan, 'Q')
df["Sex_encoded"]=pd.factorize(df["Sex"])[0]
df['Age'] = df['Age'].replace(np.nan, 29)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)

df["Embarked_encoded"]=pd.factorize(df["Embarked"])[0]
df.loc[df['Age'] <= 7.5, 'Age_encoded'] = 0
df.loc[(df['Age'] > 7.5) & (df['Age'] <= 15), 'Age_encoded'] = 1

df.loc[(df['Age'] > 15) & (df['Age'] <= 25), 'Age_encoded'] = 2
df.loc[(df['Age'] > 25) & (df['Age'] <= 30), 'Age_encoded'] = 3
df.loc[(df['Age'] > 30) & (df['Age'] <= 35), 'Age_encoded'] = 4
df.loc[(df['Age'] > 35) & (df['Age'] <= 50), 'Age_encoded'] = 5
df.loc[df['Age'] > 50, 'Age_encoded'] = 6
df["Age_encoded"].unique()

df.loc[df['Fare'] <= 12.5, 'Fare_encoded'] = 0
df.loc[(df['Fare'] > 12.5) & (df['Fare'] <= 25), 'Fare_encoded'] = 1
df.loc[(df['Fare'] > 25) & (df['Fare'] <= 50), 'Fare_encoded'] = 2
df.loc[(df['Fare'] > 50) & (df['Fare'] <= 75), 'Fare_encoded'] = 3
df.loc[(df['Fare'] > 75) & (df['Fare'] <= 100), 'Fare_encoded'] = 4
df.loc[(df['Fare'] > 100) & (df['Fare'] <= 150), 'Fare_encoded'] = 5
df.loc[df['Fare'] > 150, 'Fare_encoded'] = 6
df["Fare_encoded"].unique()
#df['Sex'] = df['Sex'].replace('female', 0)
#df['Sex'] = df['Sex'].replace('male', 1)


removed = ['Cabin','SibSp','Parch','Ticket','Name','Age','Fare','Sex','Embarked']
for x in removed:
    del df[x]

print(df.sample(15))
df.to_csv('titanic_dataset.csv')