import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

#X = dataset.drop('Survived', axis=1)
y = dataset.iloc[:, 1:2]

# splitting to train and test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 1/3, random_state = 0)

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

# TODO: don't forget to drop the Survived column when start training
