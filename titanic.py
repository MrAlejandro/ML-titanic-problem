import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

# handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset.iloc[:, 5:6])
dataset.iloc[:, 5:6] = imputer.transform(dataset.iloc[:, 5:6])

dataset["Embarked"] = dataset["Embarked"].fillna("S")

#X = dataset.drop('Survived', axis=1)
y = dataset.iloc[:, 1:2]

# splitting to train and test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 1/3, random_state = 0)
#print(X_train)

# START plot survived statistics for men and women
men_total_set = X_train[X_train['Sex'] == 'male']
men_total_qty = men_total_set.shape[0]
men_survived_qty = men_total_set[men_total_set['Survived'] == 1].shape[0]

women_total_set = X_train[X_train['Sex'] == 'female']
women_total_qty = women_total_set.shape[0]
women_survived_qty = women_total_set[women_total_set['Survived'] == 1].shape[0]

N = 2
ind = np.arange(N)
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(ind, [men_survived_qty, men_total_qty - men_survived_qty], width, color='r')
rects2 = ax.bar(ind + width, [women_survived_qty, women_total_qty - women_survived_qty], width, color='b')

ax.set_ylabel('People quantity')
ax.set_title('Survival statistics by gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Survived', 'Died'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
# END plot survived statistics for men and women

# START plot survived statistics by class
class_statistics = X_train[['Pclass', 'Survived']].groupby(['Pclass', 'Survived']).size()

N = 3
ind = np.arange(N)
width = 0.35

survived_by_class = class_statistics[1::2].values
died_by_class = class_statistics[::2].values

fig, ax = plt.subplots()
rects1 = ax.bar(ind, survived_by_class, width, color='g')
rects2 = ax.bar(ind + width, died_by_class, width, color='r')

ax.set_ylabel('People quantity')
ax.set_title('Survival statistics by class')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1st class', '2nd class', '3rd class'))

ax.legend((rects1[0], rects2[0]), ('Survived', 'Died'))

autolabel(rects1)
autolabel(rects2)

plt.show()
# END plot survived statistics by class

#START plot survived statistice by age
max_age = X_train['Age'].max()

range_step = 10
range_from = 0;
range_to = max_age + 2 * range_step

survived_by_age = X_train[X_train['Survived'] == 1][['Age', 'Survived']].groupby(pd.cut(X_train[X_train['Survived'] == 1]["Age"], np.arange(range_from, range_to, range_step))).size()
died_by_age = X_train[X_train['Survived'] == 0][['Age', 'Survived']].groupby(pd.cut(X_train[X_train['Survived'] == 0]["Age"], np.arange(range_from, range_to, range_step))).size()

N = len(survived_by_age.values)
ind = np.arange(N)
width = 0.35

fig, ax = plt.subplots(figsize=(10,5))
rects1 = ax.bar(ind, survived_by_age, width, color='g')
rects2 = ax.bar(ind + width, died_by_age, width, color='r')

ax.set_ylabel('People quantity')
ax.set_title('Survival statistics by age')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(list(survived_by_age.keys()))

ax.legend((rects1[0], rects2[0]), ('Survived', 'Died'))

autolabel(rects1)
autolabel(rects2)

plt.show()
#END plot survived statistice by age

X_train.drop(['Name', "Survived", "Cabin", "Ticket"], axis=1, inplace=True)
X_test.drop(['Name', "Survived", "Cabin", "Ticket"], axis=1, inplace=True)

# replace literal values with numeric
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit(np.unique(list(X_train['Sex'].values) + list(X_test['Sex'].values)))
X_train['Sex'] = labelencoder_X.transform(X_train['Sex'])
X_test['Sex'] = labelencoder_X.transform(X_test['Sex'])


labelencoder_X.fit(np.unique(list(X_train['Embarked'].values) + list(X_test['Embarked'].values)))
X_train['Embarked'] = labelencoder_X.transform(X_train['Embarked'])
X_test['Embarked'] = labelencoder_X.transform(X_test['Embarked'])

# perform features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
