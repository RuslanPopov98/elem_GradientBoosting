import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\xgBoost\test_1\train_features.csv')
targets = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\xgBoost\test_1\train_targets_scored.csv')

test = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\xgBoost\test_1\test_features.csv')
sub = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\xgBoost\test_1\sample_submission.csv')

df = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\titanic_set\train_data.csv')

df = pd.DataFrame(df.drop(['PassengerId'], axis=1))
df = pd.DataFrame(df.drop(['Unnamed: 0'], axis=1))
y = pd.DataFrame(df['Survived'])
X = pd.DataFrame(df.drop(['Survived'], axis=1))

X.head()

Standardize = preprocessing.StandardScaler().fit(X) #Standardization
X_stand = Standardize.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_stand, y, test_size=0.25, random_state=43)
print(X_train.shape, y_train.shape)

learning_rates = np.arange(0.05, 1, 0.1)
print(learning_rates)
training_score, valid_score = [], []

for learn_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learn_rate, max_features=2, max_depth =2)
    gb.fit(X_train, y_train)
    training_score.append(gb.score(X_train, y_train))
    valid_score.append(gb.score(X_test, y_test))

plt.plot(learning_rates, training_score, label='train')
plt.plot(learning_rates, valid_score, label='valid')

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(accuracy_score(y_test.Survived.values, y_pred))

