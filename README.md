# A-Quiet-Sleep: Understanding Chronotypes through Sleep Stages

To run the program, run the following command,
```
python3 Data_Pipeline.py
```

This will merge the FitBit datasets with the Apple Watch datasets, clean the data, perform feature engineering and create a new dataset aggregated by week.

The output screen would display the evaluation scores of the classification models used in the application, which are SVM, Random Forest, Decision Trees, and Logistic Regression classifiers.

In terms of graphs, the application displays the OvR ROC curve for the three chronotypes identified in the problem set; a plot of the sleep stages for the last 3 days in the dataset, and the distribution of sleep stages by week for the last 4 weeks in the dataset.

