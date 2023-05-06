#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, RocCurveDisplay, precision_recall_fscore_support

from sklearn.decomposition import PCA

pd.options.display.float_format = "{:,.2f}".format
df = pd.read_csv('./weatherAUS.csv')

"""
PART1: EDA
"""
print("====head====")
print(df.head(10).transpose().to_string())
print("====shape====")
print(df.shape) # (145460, 23)
print("====dtypes====")
print(df.dtypes) # 7 object, 16 float
print("====df.isnull().sum()====")
print(df.isna().sum()) # Evaporation & Sunshine have more than 40% missing

"""
DATA cleansing
"""
# drop useless features
df = df.drop(['Date', 'Location'], axis=1)
# drop more than 40% NA  features
# df = df.drop(['Sunshine', 'Evaporation'], axis=1)
print("RainTomorrow unique value", df['RainTomorrow'].unique())
df = df[~df['RainTomorrow'].isna()]
print("====remove RainTomorrow na")
print(df.isna().sum())


numerical_columns = df.select_dtypes(include='number').columns
print("====numerical_columns", numerical_columns) # 16 features

# boxplot before remove outlier
plt.figure(figsize=(15, 21))
for i, col in enumerate(numerical_columns[:-1], 1):
    plt.subplot(6, 3, i)
    sns.boxplot(data=df[col], orient="h")
    plt.title(col)
# plt.tight_layout()
# plt.title('before remove outlier')
plt.show()

# replace outlier with N/A
for feat in df.select_dtypes(include='number').columns:
    if feat == 'RainTomorrow':
        continue

    q1 = df[feat].quantile(0.25)
    q3 = df[feat].quantile(0.75)
    iqr = q3 - q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    print(feat, format(Lower_tail, '.2f'), format(Upper_tail,  '.2f'))
    print(feat, len(df[(df[feat] > Lower_tail) & (df[feat] < Upper_tail)]))
    df.loc[(df[feat] < Lower_tail) | (df[feat] > Upper_tail), feat] = np.nan

print("====after replace outlier to NA")
print(df.isna().sum())

# After remove outlier
plt.figure(figsize=(15, 15))
for i, col in enumerate(numerical_columns[:-1], 1):
    plt.subplot(6, 3, i)
    sns.boxplot(data=df[col], orient="h")
    plt.title(col)
plt.tight_layout()
plt.title("After remove outlier")
plt.show()

# fill in missing data with mode
df = df.fillna(df.mode().iloc[0])
# df = df.fillna(df.mean().iloc[0])
print("====After fill NA with mode====")
print(df.head(10).transpose().to_string())
print(df.shape) # shape is not changing
print("====isna====")
print(df.isna().sum()) # 0

print("====print transposed====")
print(df.describe().transpose().to_string())

le = preprocessing.LabelEncoder()
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])
print("feature transform df", df.head())


# pairwise plot show correlation between two features
# numerical_columns = df.select_dtypes(include='number').columns
# print("numerical_columns", numerical_columns) # 16 features
# sample = df['numerical_columns'].sample(n=1000)


# Down sampling
sample = df.sample(n=1000, random_state=5525)
print("sample", sample.info)




# ONE-HOT-ENCODING
sample = pd.get_dummies(df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])
df = sample.copy()
# print("sample", df.shape)

# plt.figure(figsize=(16, 10))
# sns.heatmap(sample.corr(), cmap='coolwarm')
# plt.show()

# Remove feature with strong correlation
# df.drop(['Temp3pm', 'Pressure9am', 'Temp9am'], axis=1, inplace=True)


# feature selection
# separate the target variable (Sales)
y = df['RainTomorrow']
X = df.drop('RainTomorrow', axis=1)


# Select columns with float data type
float_cols = X.select_dtypes(include=['float']).columns

# Standardize the values
X[float_cols] = (X[float_cols] - X[float_cols].mean()) / X[float_cols].std()

# Verify the changes
print(X.head())
# Standarlization
X_standarlization = X
print("X_standarlization", X_standarlization.head().to_string())


# split the dataset into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_standarlization,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=5525)

# display the first 5 rows of the train and test sets
print("X_train:")
print(X_train.head().to_string())
print("y_train:")
print(y_train.head().to_string())
print("X_test:")
print(X_test.head().to_string())
print("y_test:")
print(y_test.head().to_string())


# Perform backward stepwise regression analysis
selected_features = list(X.columns)
print("selected_features", selected_features)
while len(selected_features) > 0:
    # Fit a linear model using the selected features
    X_train_selected = X_train[selected_features]
    X_train_selected = sm.add_constant(X_train_selected)
    model = sm.OLS(y_train, X_train_selected).fit()
    # Find the p-value of the highest p-value feature
    pvalues = model.pvalues.drop('const')
    max_pvalue_feature = pvalues.idxmax()
    max_pvalue = pvalues.loc[max_pvalue_feature]
    # If the highest p-value feature has a p-value above the threshold, drop it from the selected features
    if max_pvalue > 0.05:
        selected_features.remove(max_pvalue_feature)
    else:
        break

# Display the eliminated and final selected features on the console
eliminated_features = set(X.columns) - set(selected_features)
print('Eliminated features:', eliminated_features)
print('Selected features:', selected_features)

# Drop the insignificant features
X_train_selected = X_train[selected_features]
# perform OLS regression analysis
model = sm.OLS(y_train, X_train_selected).fit()
# Display the OLS summary
print(model.summary())


# TODO
# y_pred = model.predict(X_test[selected_features])

# y_pred = y_pred * rain_std + rain_mu
# y_test = y_test * rain_std + rain_mu
# print(f'MSE: {mean_squared_error(y_pred=y_pred, y_true=y_test):.3f}')


# In[296]:


# PCA
df = pd.concat([X_train, y], axis=1)
# Normalization
scaler = MinMaxScaler()
X_std = scaler.fit_transform(df.select_dtypes(include=np.number))
X = X_std[:, :-1]

pca = PCA(n_components=0.9, svd_solver='full')
pca.fit(X_train)

plt.figure(figsize=(12, 8))
xticks = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)
plt.plot(xticks, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(xticks)
plt.grid()
plt.title("Perform PCA analysis")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[298]:


# Random forest
model = RandomForestRegressor(random_state=5525)

model.fit(X_train, y_train)
features = X_train.columns
importance = model.feature_importances_
selected_features_count = (importance > 0.02).sum()
indices = np.argsort(importance)[-selected_features_count:]
plt.figure()
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
selected_features = [features[i] for i in indices]
plt.yticks(range(len(indices)), selected_features)
plt.xlabel('Relative Importance regression')
plt.tight_layout()
plt.title("Random forest")
plt.show()


print("Selected: ", set(selected_features))
print("eliminated: ", set(features)-set(selected_features))


# In[300]:


# 4d
# Make a prediction on sales and compare it with the test set
# y_pred = model.predict(X_test[selected_features])
# y_pred = y_pred * rain_std + rain_mu
# # y_test = y_test * sales_std + sales_mu
#
# plt.figure()
# plt.plot(range(len(y_pred)), y_pred, label="predicted")
# plt.plot(range(len(y_test)), y_test, label="test set")
# plt.legend()
# plt.title('Prediction on test set')
# plt.ylabel('Sales')
# plt.xlabel('Observations')
# plt.show()


# In[301]:


model = RandomForestClassifier(random_state=5525, max_depth=10)
model.fit(X_train, y_train)
# features = X_train.columns.to_list()
importances = model.feature_importances_
cnt = (importances > 0.02).sum()
indices = np.argsort(importances)[-cnt:]

plt.figure()
plt.title('Random Forest Feature Importance classifier')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance classifier')
plt.tight_layout()

selected_feats_tree = [features[i] for i in indices]

len(selected_feats_tree)



# Phase II: backward stepwise regression analysis
import statsmodels.api as sm

def backward_stepwise(X, y, cols):
    del_features = []
    while True:
        x = X[cols]
        results = sm.OLS(y, x).fit()
        p_sorted = results.pvalues.sort_values()

        if p_sorted[-1] < 0.05:
            break

        cols.remove(p_sorted.index[-1])
        del_features.append(p_sorted.index[-1])

    return cols, del_features



selected_feats_sw, del_features = backward_stepwise(X_train, y_train, X_train.columns.to_list())
x = X_train[selected_feats_sw]
results = sm.OLS(y_train, x).fit()
print(results.summary())



print(len(selected_feats_sw))

intersection = list(set(selected_feats_tree) & set(selected_feats_sw))

print(len(intersection))
print(intersection)


# In[305]:


# Phase III: Classification analysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, f1_score, precision_recall_fscore_support
import time

start = time.time()
estimator = LogisticRegression(random_state=5525)
tuned_parameters = [{
    'C': [0.2, 0.5, 1.0, 1.5, 2.0],
    'max_iter': [50, 100, 200, 300]
}]

grid_search = GridSearchCV(estimator=estimator, param_grid=tuned_parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train[intersection], y_train.values.ravel())
print(f'best_params_: {grid_search.best_params_}')
clf_best = LogisticRegression(**grid_search.best_params_)
clf_best.fit(X_train[intersection], y_train.values.ravel())

predictions = clf_best.predict_proba(X_test[intersection])
y_predicted = clf_best.predict(X_test[intersection])

end = time.time()
print(f'Logistic regression time: {end - start}')


precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted,average="micro")
# print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
# print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
print(precision, recall, f1)
print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
print(f'Confusion matrix of 9: {confusion_matrix(y_test,  y_predicted)}')


print(classification_report(y_test, y_predicted, target_names=['0', '1']))
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic regression confusion_matrix')
plt.show()


## DesicionTree
start = time.time()

estimator = DecisionTreeClassifier(random_state=5525)
tuned_parameters = [{
    'max_depth': [5, 10, 15, 20, 30],
}]

grid_search = GridSearchCV(estimator=estimator, param_grid=tuned_parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train[intersection], y_train.values.ravel())
print(f'best_params_: {grid_search.best_params_}')
clf_best = DecisionTreeClassifier(**grid_search.best_params_)
clf_best.fit(X_train[intersection], y_train.values.ravel())

predictions = clf_best.predict_proba(X_test[intersection])
y_predicted = clf_best.predict(X_test[intersection])

end = time.time()
print(f'Decision tree time: {end - start}')

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted, average="micro",)
# print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
print(f'Confusion matrix of 9: {confusion_matrix(y_test, y_predicted)}')


print(classification_report(y_test, y_predicted, target_names=['0', '1']))
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Decision tree confusion_matrix')
plt.show()


## KNN
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

estimator = KNeighborsClassifier()
tuned_parameters = [{
    'n_neighbors': [5, 10, 15, 30],
}]

grid_search = GridSearchCV(estimator=estimator, param_grid=tuned_parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train[intersection], y_train.values.ravel())
print(f'best_params_: {grid_search.best_params_}')
clf_best = KNeighborsClassifier(**grid_search.best_params_)
clf_best.fit(X_train[intersection], y_train.values.ravel())

predictions = clf_best.predict_proba(X_test[intersection])
y_predicted = clf_best.predict(X_test[intersection])

end = time.time()
print(f'KNN time: {end - start}')

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted, average="micro",)
# print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
print(f'Confusion matrix of 9: {confusion_matrix(y_test, y_predicted)}')


print(classification_report(y_test, y_predicted, target_names=['0', '1']))
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN confusion_matrix')
plt.show()


## Naive bayesian
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

clf_best = GaussianNB()
clf_best.fit(X_train[intersection], y_train.values.ravel())

predictions = clf_best.predict_proba(X_test[intersection])
y_predicted = clf_best.predict(X_test[intersection])

end = time.time()
print(f'Naive bayesian time: {end - start}')

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted, average="micro",)
# print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
print(f'Confusion matrix of 9: {confusion_matrix(y_test,  y_predicted)}')
print(classification_report(y_test, y_predicted, target_names=['0', '1']))
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Naive bayesian confusion_matrix')
plt.show()




## RandomForest
start = time.time()

estimator = RandomForestClassifier(random_state=5525)
tuned_parameters = [{
    'n_estimators': [50,100,200],
    'max_depth':[5, 15, 30]
}]

grid_search = GridSearchCV(estimator=estimator, param_grid=tuned_parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train[intersection], y_train.values.ravel())
print(f'best_params_: {grid_search.best_params_}')
clf_best = RandomForestClassifier(**grid_search.best_params_)
clf_best.fit(X_train[intersection], y_train.values.ravel())

predictions = clf_best.predict_proba(X_test[intersection])
y_predicted = clf_best.predict(X_test[intersection])

end = time.time()
print(f'Random forest time: {end - start}')

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted, average="micro",)
# print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
print(f'Confusion matrix of 9: {confusion_matrix(y_test, y_predicted)}')

print(classification_report(y_test, y_predicted, target_names=['0', '1']))
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random forest confusion_matrix')
plt.show()



## NN
# from keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Dense
# import tensorflow as tf
#
# start = time.time()
# ANN_model = Sequential()
# ANN_model.add(InputLayer(input_shape=(7, )))
#
# ANN_model.add(Dense(10, activation='sigmoid'))
# ANN_model.add(Dense(1, activation='sigmoid'))
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
#
# ANN_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = ANN_model.fit(X_train[intersection], y_train, epochs=10, batch_size=128, validation_split=0.2, shuffle=False)
#
#
# y_predicted = np.argmax(predictions, axis=1)
#
# end = time.time()
# print(f'Neural network  time: {end - start}')
#
# precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predicted, average="micro",)
# # print(f'accuracy_score of 9: {accuracy_score(y_test, y_predicted):.2f}')
# # print(f'f1_score of 9: {f1_score(y_test, y_predicted, average="micro",):.2f}')
# print(f'precision: {precision:.2f} recal: {recall:.2f} f1: {f1:.2f}')
# print(f'AUC of 9: {roc_auc_score(y_test, predictions[:, 1]):.2f}')
# print(f'Confusion matrix of 9: {confusion_matrix(y_test, y_predicted)}')
#
#
# print(classification_report(y_test, y_predicted, target_names=['0', '1']))
# cm = confusion_matrix(y_test, y_predicted)
# sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Neural network confusion_matrix')
# plt.show()


## phase 4
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
  
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X_train[intersection])
  
    distortions.append(sum(np.min(cdist(X_train[intersection], kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X_train[intersection].shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(X_train[intersection], kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X_train[intersection].shape[0]
    mapping2[k] = kmeanModel.inertia_


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

