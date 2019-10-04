# Question 1. (Holdout)
# Create a training set that contains 80% of the labeled data and export it to a .csv file called
# training.csv. Create a test set that contains the remaining 20% and export it to a .csv file called
# testing.csv. Submit the two .csv files


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('csv/glass.csv', header=None)

data.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

print(data)
Y = data['Type']
X = data.drop(['Id', 'Type'], axis=1)
print(X)
print(Y)

#########################################
# Training and Test set creation
#########################################


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

np.savetxt('csv/X_train.csv', X_train, delimiter=', ')
np.savetxt('csv/X_test.csv', X_test, delimiter=', ')
np.savetxt('csv/Y_train.csv', Y_train, delimiter=', ')
np.savetxt('csv/Y_test.csv', Y_test, delimiter=', ')

# Question 2. (Decision Tree Classifier)
# Use the training and test sets in Question 1, perform the following tasks:
# 2a. Using entropy as the impurity measure for splitting criterion, fit decision trees of different
# maximum depths [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25] to the training set. Submit the plot showing
# their respective training and test accuracies when applied to the training and test sets. What do you
# find?


from sklearn import tree

print('Use Entropy index for impurity measure :')
accuracy = np.empty(12, dtype=float)
max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
i = 0
for max_depth in max_depths:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)

    import pydotplus

    # create graph tree without class names
    # dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['1', '2', '3', '4', '5', '6', '7'],
    #                                 filled=True,
    #                                 out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_png('image/tree.png')

    # create graph tree with class names

    dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['building_windows_float_processed',
                                                                               'nonbuilding_windows_non_float_processed',
                                                                               'vehicle_windows_float_processed',
                                                                               'vehicle_windows_non_float_processed',
                                                                               'containers', 'tableware', 'headlamps'],
                                    filled=True,
                                    out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('image/tree-entropy-%d.png' % (max_depth))

    predY = clf.predict(X_test)
    predictions = pd.concat([data['Id'], pd.Series(predY, name='Predicted Class')], axis=1)

    from sklearn.metrics import accuracy_score

    accuracy[i] = accuracy_score(Y_test, predY)

    print(' Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy[i])))
    i += 1
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_prop_cycle(color=['red'])
# ax[0].set_ylim([0, 0.8])
ax[0].plot(max_depths, accuracy)
ax[0].legend(['accuracy-Entropy'], loc='upper left')

# plt.show()
#
# 2b. Using Gini index as impurity measure, fit decision trees of different maximum depths [2, 3, 4, 5,
# 6, 7, 8, 9, 10, 15, 20, 25] to the training set. Submit the plot showing their respective training and
# test accuracies when applied to the training and test sets. What do you find?
i = 0
print('Use Gini index for impurity measure :')
for max_depth in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]:
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)

    import pydotplus

    # create graph tree without class names
    # dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['1', '2', '3', '4', '5', '6', '7'],
    #                                 filled=True,
    #                                 out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_png('image/tree.png')

    # create graph tree with class names

    dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['building_windows_float_processed',
                                                                               'non-building_windows_non_float_processed',
                                                                               'vehicle_windows_float_processed',
                                                                               'vehicle_windows_non_float_processed',
                                                                               'containers', 'tableware', 'headlamps'],
                                    filled=True,
                                    out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('image/tree-gini-%d.png' % (max_depth))

    predY = clf.predict(X_test)
    predictions = pd.concat([data['Id'], pd.Series(predY, name='Predicted Class')], axis=1)

    from sklearn.metrics import accuracy_score

    accuracy[i] = accuracy_score(Y_test, predY)
    print(' Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy_score(Y_test, predY))))
    i += 1

ax[1].set_prop_cycle(color=['green'])
# ax[1].set_ylim([0, 0.8])
ax[1].plot(max_depths, accuracy)
ax[1].legend(['accuracy-Gini'], loc='upper left')
plt.show()


# Question 3. (k-Nearest Neighbor Classifier)
# Use the training and test sets in Question 1, train a k-nearest neighbor classifier and measure
# performance on both the training set and the test set. Vary the settings in the following ways:
# • Try it for k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
# • Try it with Euclidean distance, Manhattan distance, and cosine distance.
# Submit plots showing the trend as k varies from 1 to 25 for each of the three distances, focusing on
# both training set and test set accuracies. What do you find?

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

numNeighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]

# Manhattan distance
trainAcc = []
testAcc = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, metric='manhattan', p=1)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc.append(accuracy_score(Y_train, Y_predTrain))
    testAcc.append(accuracy_score(Y_test, Y_predTest))

plt.clf()
plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Manhattan distance')
plt.show()

# Euclidean distance
trainAcc = []
testAcc = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=2)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc.append(accuracy_score(Y_train, Y_predTrain))
    testAcc.append(accuracy_score(Y_test, Y_predTest))

plt.clf()
plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Euclidean distance')
plt.show()

# Cosine distance

import sklearn.metrics.pairwise as smp
trainAcc = []
testAcc = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc.append(accuracy_score(Y_train, Y_predTrain))
    testAcc.append(accuracy_score(Y_test, Y_predTest))

plt.clf()
plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Cosine distance')
plt.show()