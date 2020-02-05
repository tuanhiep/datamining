import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1a. For each quantitative attribute, calculate its average, standard deviation, minimum, and maximum
# values.
data = pd.read_csv('csv/BreastCancerCoimbra.csv', header="infer")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(data.head(10))
# data = data.convert_objects(convert_numeric=True)
from pandas.api.types import is_numeric_dtype

for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())

#  Another way to describe the summary statistics of data set
data = data.iloc[1:]
data.columns = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1',
                'Classification']
print(data.describe(include='all'))

# 1b. Compute the covariance and correlation between pairs of attributes

print('Covariance:')
print(data.cov())

# 1c. Display the histogram for each of the quantitative attributes by discretizing it into 10 separate bins
# and counting the frequency for each bin.


fig, axes = plt.subplots(nrows=2, ncols=5)
# This array is to store axes of the figure
axs = np.empty(10, dtype=object)
for i, ax in enumerate(axes.flat):
    axs[i] = ax
count = 0
for col in data.columns:
    if is_numeric_dtype(data[col]):
        axs[count].hist(data[col], 10)
        axs[count].set_title(col)
        count += 1
plt.tight_layout()
plt.show()

# 1d. Display a boxplot to show the distribution of values for each attribute. Which attribute has outliers?
# Clear previous setting for pyplot
plt.clf()
# Show boxplot
plt.show(data.boxplot())

# 1e. Consider the first four attributes: Age (years), BMI (kg/m2), Glucose (mg/dL), Insulin (µU/mL). For each pair of
# those four attributes, use a scatter plot to visualize their joint distribution. Based on the scatter plot, what are
# possible correlations that you can observe?

plt.clf()
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
index = 0
for i in range(3):
    for j in range(i + 1, 4):
        ax1 = int(index / 2)
        ax2 = index % 2
        axes[ax1][ax2].scatter(data[data.columns[i]], data[data.columns[j]], color='red')
        axes[ax1][ax2].set_xlabel(data.columns[i])
        axes[ax1][ax2].set_ylabel(data.columns[j])
        index = index + 1
plt.show()

# 1f.Use parallel coordinates to visualize the dataset.The visualization should have a legend that shows different
# labels
plt.clf()

from pandas.plotting import parallel_coordinates


fig, ax = plt.subplots()
parallel_coordinates(data, 'Classification')
plt.legend(['Classification 1', 'Classification 2'], loc='upper left')
plt.show()

# Question 2. (Data Preprocessing)
# Use the same data set as in Question 1. Write programs to perform the following tasks.
# 2a. Create a sample of size 10 which is randomly selected (without replacement) from the original
# data.
sample = data.sample(n=10)
print(sample)

