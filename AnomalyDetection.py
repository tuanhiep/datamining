# Question 1. (Local Outlier Factor)
# Remove rows with missing values. Perform unsupervised outlier detection using Local Outlier
# Factor (LOF) with number of neighbors = 10 and metric = Euclidean distance. Use default values
# for other parameters. Plot the ROC curve. Report the processing time and AUC.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import  roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from time import time
df = pd.read_csv('csv/breast-cancer-wisconsin.csv', header=None)
data_raw = df.replace('?', np.NaN)
data = data_raw.dropna()
data.columns = ['Id', 'Clump-Thickness', 'Uniformity-of-Cell-Size', 'Uniformity-of-Cell-Shape', 'Marginal-Adhesion',
                'Single-Epithelial-Cell-Size', 'Bare-Nuclei', 'Bland-Chromatin', ' Normal-Nucleoli', 'Mitoses', 'Class']

data['Class'] = data['Class'].replace([2], 1)
data['Class'] = data['Class'].replace([4], -1)

Y = data['Class']
X = data.drop(['Id', 'Class'], axis=1)

ground_truth = Y.copy()
print(ground_truth)

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=10, metric='euclidean', contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
tstart = time()
clf.fit_predict(X)
y_pred = clf.negative_outlier_factor_
processing_time = time() - tstart

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(ground_truth, y_pred)
plt.plot(false_positive_rate, true_positive_rate)
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

print("Local Outlier Factor has processing time = {} and  AUC = {}".format(processing_time, roc_auc_score(ground_truth, y_pred)))


# Question 2. (Isolation Forest)
# Remove rows with missing values. Perform unsupervised outlier detection using Isolation Forest
# with number of trees = 100, sub-sampling size = 256. Use default values for other parameters. Plot
# the ROC curve. Report the processing time and AUC.

from sklearn.ensemble import IsolationForest

clf = IsolationForest(behaviour='new', max_samples=256,
                      random_state=0, contamination='auto')
tstart = time()
clf.fit(X)
y_pred = clf.score_samples(X)
processing_time = time() - tstart

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(Y, y_pred)
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
print("Isolation Forest has processing time = {} and AUC = {}".format(processing_time, roc_auc_score(ground_truth, y_pred)))
