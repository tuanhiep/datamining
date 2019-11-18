import numpy as np
import pandas as pd
from sklearn import cluster, metrics, ensemble
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
import pydotplus
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('csv/kidneyclean.csv')
print('Number of rows in original data = %d' % (data.shape[0]))
print(data)
best_attributes = ["wbcc", "bu", "bgr", "al", "sc", "pcv", "su", "htn", "pc", "dm", "class"]
drop_list = [i for i in data.columns if i not in best_attributes]
data.drop(drop_list, axis=1, inplace=True)
print(data)
# separate the data from the target attributes
Y = data['class']
X_before_normalized = data.drop(['class'], axis=1)
# normalize the data attributes
X = preprocessing.normalize(X_before_normalized)
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

# Decision Tree Classifier

#  Using entropy as the impurity measure for splitting criterion, fit decision trees of different
#  maximum depths [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25] to the training set

print('Use Entropy index for impurity measure :')
accuracy = np.empty(12, dtype=float)
max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
i = 0
for max_depth in max_depths:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)
    # create graph tree with class names
    dot_data = tree.export_graphviz(clf, feature_names=X_before_normalized.columns, class_names=['1', '0'], filled=True,
                                    out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('image/tree-entropy-%d.png' % (max_depth))
    predY = clf.predict(X_test)
    accuracy[i] = accuracy_score(Y_test, predY)
    print(' Entropy: Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy[i])))
    i += 1
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_prop_cycle(color=['red'])
ax[0].set_ylim([0.8, 1.1])
ax[0].plot(max_depths, accuracy)
ax[0].legend(['accuracy-Entropy'], loc='upper left')

#  Using Gini index as impurity measure, fit decision trees of different maximum depths [2, 3, 4, 5,
#  6, 7, 8, 9, 10, 15, 20, 25] to the training set
i = 0
print('Use Gini index for impurity measure :')
for max_depth in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]:
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)
    dot_data = tree.export_graphviz(clf, feature_names=X_before_normalized.columns, class_names=['1', '0'],
                                    filled=True,
                                    out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('image/tree-gini-%d.png' % (max_depth))
    predY = clf.predict(X_test)
    accuracy[i] = accuracy_score(Y_test, predY)
    print('Gini: Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy_score(Y_test, predY))))
    i += 1

ax[1].set_prop_cycle(color=['green'])
ax[1].set_ylim([0.8, 1.1])
ax[1].plot(max_depths, accuracy)
ax[1].legend(['accuracy-Gini'], loc='upper left')
plt.show()

# k-Nearest Neighbor Classifier

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
    print("K= %d - mean accuracy : %f" % (k, scores.mean()))

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

# Logistic regression Classifier
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4, shuffle=True)
np.savetxt('csv/X_train.csv', X_train, delimiter=', ')
np.savetxt('csv/X_test.csv', X_test, delimiter=', ')
np.savetxt('csv/Y_train.csv', Y_train, delimiter=', ')
np.savetxt('csv/Y_test.csv', Y_test, delimiter=', ')
from sklearn import linear_model

plt.clf()
C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
i = 1
for param in C:
    log_reg = linear_model.LogisticRegression(C=param, solver='lbfgs')
    log_reg.fit(X_train, Y_train)
    Y_predict_test = log_reg.predict(X_test)
    y_predict_prob = log_reg.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Y_test, y_predict_prob)
    plt.subplot(2, 5, i)
    plt.plot(false_positive_rate, true_positive_rate, i)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('ROC curve C = {}'.format(param))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    # Use score method to get accuracy of model
    score = log_reg.score(X_test, Y_test)
    print("Logistic Regression with C = %f give us the accuracy: %f" % (param, score))
    i = i + 1
plt.show()


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', true_positive_rate[thresholds > threshold][-1])
    print('Specificity:', 1 - false_positive_rate[thresholds > threshold][-1])


evaluate_threshold(0.5)

# Support Vector Machine Classifier
plt.clf()
C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
SVM_train_accuracy = []
SVM_test_accuracy = []

for param in C:
    clf = SVC(C=param, kernel='linear')
    clf.fit(X_train, Y_train)
    Y_predict_SVM_train = clf.predict(X_train)
    Y_predict_SVM_test = clf.predict(X_test)
    SVM_train_accuracy.append(accuracy_score(Y_train, Y_predict_SVM_train))
    SVM_test_accuracy.append(accuracy_score(Y_test, Y_predict_SVM_test))
    print("Support Vector Machine  with C = %f give us the train accuracy: %f and test accuracy: %f " % (
        param, accuracy_score(Y_train, Y_predict_SVM_train), accuracy_score(Y_test, Y_predict_SVM_test)))

plt.plot(C, SVM_train_accuracy, 'ro-')
plt.plot(C, SVM_test_accuracy, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('C')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.title('Support Vector Machine')
plt.show()

# Nonlinear Support Vector Machine

C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
NL_SVM_train_Accuracy = []
NL_SVM_test_Accuracy = []

for param in C:
    clf = SVC(C=param, kernel='rbf', gamma='auto')
    clf.fit(X_train, Y_train)
    Y_predict_NL_SVMTrain = clf.predict(X_train)
    Y_predict_NL_SVM_Test = clf.predict(X_test)
    NL_SVM_train_Accuracy.append(accuracy_score(Y_train, Y_predict_NL_SVMTrain))
    NL_SVM_test_Accuracy.append(accuracy_score(Y_test, Y_predict_NL_SVM_Test))
    print("Nonlinear Support Vector Machine  with C = %f give us the train accuracy: %f and test accuracy: %f " % (
        param, accuracy_score(Y_train, Y_predict_NL_SVMTrain), accuracy_score(Y_test, Y_predict_NL_SVM_Test)))
plt.plot(C, NL_SVM_train_Accuracy, 'ro-')
plt.plot(C, NL_SVM_test_Accuracy, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('C')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.title('Nonlinear Support Vector Machine')
plt.show()

# Ensemble Methods
numBaseClassifiers = 500
max_depth_EM = 10
train_Acc = []
test_Acc = []
## Random Forest Classifier
clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predict_train_EM = clf.predict(X_train)
Y_predict_test_EM = clf.predict(X_test)
train_Acc.append(accuracy_score(Y_train, Y_predict_train_EM))
test_Acc.append(accuracy_score(Y_test, Y_predict_test_EM))
print("Ensemble Method by Random Forest Classifier give us train accuracy: %f and test accuracy: %f " % (
    accuracy_score(Y_train, Y_predict_train_EM), accuracy_score(Y_test, Y_predict_test_EM)))
## Bagging Classifier
clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth_EM), n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predict_train_EM = clf.predict(X_train)
Y_predict_test_EM = clf.predict(X_test)
train_Acc.append(accuracy_score(Y_train, Y_predict_train_EM))
test_Acc.append(accuracy_score(Y_test, Y_predict_test_EM))
print("Ensemble Method by Bagging Classifier give us train accuracy: %f and test accuracy: %f " % (
    accuracy_score(Y_train, Y_predict_train_EM), accuracy_score(Y_test, Y_predict_test_EM)))
## Adaboost Classifier
clf = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth_EM), n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predict_train_EM = clf.predict(X_train)
Y_predict_test_EM = clf.predict(X_test)
train_Acc.append(accuracy_score(Y_train, Y_predict_train_EM))
test_Acc.append(accuracy_score(Y_test, Y_predict_test_EM))
print("Ensemble Method by Adaboost Classifier give us train accuracy: %f and test accuracy: %f " % (
    accuracy_score(Y_train, Y_predict_train_EM), accuracy_score(Y_test, Y_predict_test_EM)))

methods = ['Random Forest', 'Bagging', 'AdaBoost']
plt.plot(methods, train_Acc, 'ro-')
plt.plot(methods, test_Acc, 'bv--')
plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.title("Ensemble Method")
plt.show()
