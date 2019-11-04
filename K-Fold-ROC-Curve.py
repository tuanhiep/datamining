# Question 1. (k-NN Classifier and 5-fold cross validation)
# We want to train a k-NN Classifier for the Iris Data Set. Use 5-fold cross validation to select a good k
# from [1,50] for the k-NN Classifier. Submit k and the source code.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv('csv/iris.csv', header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
Y = data['Class']
X = data.drop(['Class'], axis=1)
# We can create k-fold cross validation sets by using KFold from sklearn.model_selection, then we calculate the accuracy
# for each cross-validation set, then we calculate the mean accuracy for each number of neighbors K as the previous
# assignment but this approach is slower than using the provided cross_val_score function
# kf = KFold(n_splits=5)
# print(kf)
# for train_index, test_index in kf.split(X):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]


# Use the provided cross_val_score function  of scikit-learn
# creating list of K for KNN from 1 to 50
neighbors = list(range(1, 50))

# empty list that will hold mean accuracy for each number of neighbor k
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# we calculate the mis-Classification error from the accuracy
mse = [1 - x for x in cv_scores]

# Best number of K resulting the lowest mis-Classification error (highest mean accuracy)
best_k = neighbors[mse.index(min(mse))]
print("The best number of neighbors is {}".format(best_k))

# plot mis-Classification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Mis-Classification Error")
plt.show()

# Question 2. (ROC Curve)
# Convert the problem in Iris Data Set into a binary Classification task (setosa versus non-setosa). We
# can do so by replacing the Class labels of the instances to non-setosa except for those that belong
# to the setosa Class.
# Create a training set that contains 80% of the labeled data and create a test set that contains the
# remaining 20%. Train a logistic regression Classifier using the training set. Plot the ROC curve for
# the setosa Class (positive Class) when applying the logistic regression Classifier to the test set.
# Submit the plot and the source code.
plt.clf()
data['Class'] = data['Class'].replace(['Iris-virginica', 'Iris-versicolor'], 0)
data['Class'] = data['Class'].replace(["Iris-setosa"], 1)
Y = data['Class']
X = data.drop(['Class'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4, shuffle=True)
np.savetxt('csv/X_train.csv', X_train, delimiter=', ')
np.savetxt('csv/X_test.csv', X_test, delimiter=', ')
np.savetxt('csv/Y_train.csv', Y_train, delimiter=', ')
np.savetxt('csv/Y_test.csv', Y_test, delimiter=', ')
from sklearn import linear_model

C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
# C = [0.01, 0.1]
i=1
for param in C:
    log_reg = linear_model.LogisticRegression(solver='lbfgs')
    log_reg.fit(X_train, Y_train)
    Y_predict_test = log_reg.predict(X_test)
    y_predict_prob = log_reg.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Y_test, y_predict_prob)
    plt.subplot(2,5,i)
    plt.plot(false_positive_rate, true_positive_rate,i)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('ROC curve C = {}'.format(param))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    i=i+1
plt.show()


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', true_positive_rate[thresholds > threshold][-1])
    print('Specificity:', 1 - false_positive_rate[thresholds > threshold][-1])

evaluate_threshold(0.5)