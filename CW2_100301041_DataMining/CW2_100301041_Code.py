#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[2]:


#Importing the data csv
#data = pd.read_csv('LoanStats_2017Q2.csv')
#a data error was thrown when attempting to read in the full dataset
data = pd.read_csv('data_31stmarch.csv')
d_count = data.count()


# In[3]:


try:
    assert data.loan_amnt.count() == 50000
    print("Expected file size is correct")
except:
    print("Incorrect file size")
#checking the correct number of rows were read in


# In[4]:


#dropping member_id as all rows were empty
data = data.drop(columns = ['member_id']) 
#dropping all columns that are too logically similar.
data = data.drop(columns = ['funded_amnt', 'funded_amnt_inv', 'installment'])
#too similar to loan_amnt
data = data.drop(columns = ['total_pymnt_inv', 'collection_recovery_fee','sub_grade', 'open_il_12m', 'out_prncp_inv'])
#dropping columns with greater than 60% of data missing or nan
data = data.dropna(thresh = data.shape[0]*0.6, how='all',axis=1)
#dropping other columns that are not necessary, want to make the model as "light" as possible
data = data.drop(columns = ['issue_d', 'addr_state','pymnt_plan','verification_status', 'last_credit_pull_d','initial_list_status'])
print(data.head())


# In[5]:


#filling emp_length nan with 0 (as no employment length must mean 0 years in employment)
data.emp_length = data.emp_length.fillna(0)
data = data.dropna()


# In[6]:


#One-hot encoding all categorical columns so that they can be utilised for classification
data_home_ownership_onehot = pd.get_dummies(data.home_ownership, prefix='home_ownership')
#print(data_home_ownership_onehot.head())
data_grade_onehot = pd.get_dummies(data.grade, prefix ='grade')
data_application_type_onehot = pd.get_dummies(data.application_type, prefix = 'application_type')
data = data.join([data_home_ownership_onehot, data_grade_onehot, data_application_type_onehot])


# In[7]:


#replacing loan_status strings with integers (Current = 0, Fully Paid = 1, Late(16-30 days) = 2, Late(31-120 days) = 3)
data.loan_status = data.loan_status.replace(['Current','Fully Paid', 'Late(16-30 days)', 'Late(31-120 days)'], ['0','1','2','3'])
#loan status set to be either late(1) or not(0)
data.loan_status = data.loan_status.replace(['0','1'], '0')
data.loan_status = data.loan_status.replace(['2','3'], '1')
#print(data.loan_status)
#This data can now be used as labels for the classifier


# In[8]:


#stripping non-number values from cells
#removing 'months' from the term column to just give an int
data.term = data.term.str.replace(r'\D+', '')
#removing years, +, < from emp_length
data.emp_length = data.emp_length.str.replace(r'\D+', '')
#filling nan in emp_length with 0
data.emp_length = data.emp_length.fillna(0)
#trimming % symbols from int_rate and revol_util
data.int_rate = data.int_rate.str.replace(r'%', '')
data.revol_util = data.revol_util.str.replace(r'%', '')
#print(data.revol_util)


# In[9]:


#creating the labels for training_data
labels = data.loan_status
#print(labels_indiv)
#dropping categorical features
training_data = data.drop(columns = ['home_ownership','purpose', 'application_type', 'loan_status', 'grade'], axis = 1)


# In[10]:


#attempting to build a "light" model - processing data 

light_data = data.copy()
light_data = light_data.loc[:, ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'pub_rec', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'pub_rec_bankruptcies', 'loan_status', 'home_ownership_ANY','home_ownership_MORTGAGE', 
        'home_ownership_NONE', 'home_ownership_OWN','home_ownership_RENT', 
        'grade_A', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'application_type_ ',
       'application_type_DIRECT_PAY', 'application_type_INDIVIDUAL',
       'application_type_JOINT']]

light_data_labels = light_data.loan_status
light_data = light_data.drop(columns = ['loan_status'])

#using the best split stated earlier - 60% training data
light_train_data, light_test_data, light_train_labels, light_test_labels = train_test_split(light_data, light_data_labels, train_size = 0.6, random_state = 6)


# In[11]:


#creating a train_test_split so that validation can be done after training.
train_data, test_data, train_labels, test_labels = train_test_split(training_data, labels, train_size = 0.8, random_state = 33)


# In[12]:


#defining and training classifier
import time

start_1_train = time.time()
classifier_1 = SVC(probability=True)
classifier_1.fit(train_data, train_labels)
run_time_1_train = time.time() - start_1_train
print("train time:", run_time_1_train, "seconds")


# In[13]:


#classifier accuracy for the data_set (rbf kernel)
start_1_test = time.time()
print("Classification accuracy with an rbf kernel:", classifier_1.score(test_data, test_labels))
run_time_1_test = time.time() - start_1_test
print(run_time_1_test, "seconds")


# In[14]:


#creating and training classifier with a poly kernel
start_2_train = time.time()
classifier_2 = SVC(kernel = 'poly')
classifier_2.fit(train_data, train_labels)
run_time_2_train = time.time() - start_2_train
print("train time:", run_time_2_train, "seconds")
#classifier accuracy for a poly kernel
start_2_test = time.time()
print("Classification accuracy with a poly kernel", classifier_2.score(test_data, test_labels))
run_time_2_test = time.time() - start_2_test
print(run_time_2_test, "seconds")


# In[ ]:


#creating and training classifier with a linear kernel
start_3_train = time.time()
classifier_3 = SVC(kernel = 'linear')
classifier_3.fit(train_data, train_labels)
run_time_3_train = time.time() - start_3_train
print("train time:", run_time_3_train, "seconds")
#classifier accuracy for a linear kernel
start_3_test = time.time()
print("Classification accuracy with a linear kernel", classifier_3.score(test_data, test_labels))
run_time_3_test = time.time() - start_3_test
print(run_time_3_test, "seconds")


# In[15]:


from sklearn.metrics import f1_score, precision_score, recall_score
#only test data and labels will be used, train_test_split called to get a random selection.
filler_train, metrics_test_data, filler_labels, metrics_test_labels = train_test_split(training_data, labels, train_size = 0.5, random_state = 46)

#getting these metrics for classifer 1 (rbf)
start_1_predict = time.time()
predicted_labels_1 = classifier_1.predict(metrics_test_data)
run_time_1_predict = time.time() - start_1_predict
print("Prediction time:", run_time_1_predict, "seconds")

f1_classifier_1 = f1_score(metrics_test_labels, predicted_labels_1, average = 'micro')
precision_classifier_1 = precision_score(metrics_test_labels, predicted_labels_1, average = 'micro')
recall_classifier_1 = recall_score(metrics_test_labels, predicted_labels_1, average = 'micro')

print("F1 score for classifier 1 (rbf kernel):", f1_classifier_1)
print("Precision score for classifier 1 (rbf kernel):", precision_classifier_1)
print("Recall score for classifier 1 (rbf kernel):", recall_classifier_1)


# In[16]:


#getting metrics imported above for classifier 2 (poly)
start_2_predict = time.time()
predicted_labels_2 = classifier_2.predict(metrics_test_data)
run_time_2_predict = time.time() - start_2_predict
print("Prediction time:", run_time_2_predict, "seconds")

f1_classifier_2 = f1_score(metrics_test_labels, predicted_labels_2, average = 'micro')
precision_classifier_2 = precision_score(metrics_test_labels, predicted_labels_2, average = 'micro')
recall_classifier_2 = recall_score(metrics_test_labels, predicted_labels_2, average = 'micro')

print("F1 score for classifier 2 (poly kernel):", f1_classifier_2)
print("Precision score for classifier 2 (poly kernel):", precision_classifier_2)
print("Recall score for classifier 2 (poly kernel):", recall_classifier_2)


# In[ ]:


#getting metrics imported above for classifier 3 (linear)
start_3_predict = time.time()
predicted_labels_3 = classifier_3.predict(metrics_test_data)
run_time_3_predict = time.time() - start_3_predict
print("Prediction time:", run_time_3_predict, "seconds")

f1_classifier_3 = f1_score(metrics_test_labels, predicted_labels_3, average = 'micro')
precision_classifier_3 = precision_score(metrics_test_labels, predicted_labels_3, average = 'micro')
recall_classifier_3 = recall_score(metrics_test_labels, predicted_labels_3, average = 'micro')

print("F1 score for classifier 3 (linear kernel):", f1_classifier_3)
print("Precision score for classifier 3 (linear kernel):", precision_classifier_3)
print("Recall score for classifier 3 (linear kernel):", recall_classifier_3)


# In[17]:


#trying different test-train ratios

splits = [0.5, 0.6, 0.7, 0.8, 0.9]

f1_scores = []
accuracy_scores = []
classifier_split_test = SVC()
for i in range(len(splits)):
    train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(training_data, labels, train_size = splits[i], random_state = 26)
    classifier_split_test.fit(train_data_split, train_labels_split)
    predictions = classifier_split_test.predict(metrics_test_data)
    f1_scores.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
    accuracy_scores.append(classifier_split_test.score(test_data_split, test_labels_split))
print(f1_scores)
print(accuracy_scores)


# In[18]:


#generating 10 random numbers for different random states
import random

randstate = []

for i in range(0,10):
    x = random.randint(1,100)
    randstate.append(x)
print(randstate)


# In[19]:


#instantiating all lists for success metrics
#creating empty lists for each number in splits(50 = 50%)
f1_scores_50 = []
f1_scores_60 = []
f1_scores_70 = []
f1_scores_80 = []
f1_scores_90 = []

accuracy_scores_50 = []
accuracy_scores_60 = []
accuracy_scores_70 = []
accuracy_scores_80 = []
accuracy_scores_90 = []


# In[21]:


#testing the combinations of random_states and different test-train splits
#creating empty lists for each number in splits(50 = 50%)

#looping through every instance in splits for every instance in randstate, giving arrays of
#the f1_score for every combination of random state and train-test split
for rs in range(len(randstate)):
    for i in range(len(splits)):
        train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(training_data, labels, train_size = splits[i], random_state = randstate[rs])
        classifier_split_test.fit(train_data_split, train_labels_split)
        predictions = classifier_split_test.predict(metrics_test_data)
        if splits[i] == 0.5:
            f1_scores_50.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
            accuracy_scores_50.append(classifier_split_test.score(test_data_split, test_labels_split))
        elif splits[i] == 0.6:
            f1_scores_60.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
            accuracy_scores_60.append(classifier_split_test.score(test_data_split, test_labels_split))
        elif splits[i] == 0.7:
            f1_scores_70.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
            accuracy_scores_70.append(classifier_split_test.score(test_data_split, test_labels_split))
        elif splits[i] == 0.8:
            f1_scores_80.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
            accuracy_scores_80.append(classifier_split_test.score(test_data_split, test_labels_split))
        elif splits[i] == 0.9:
            f1_scores_90.append(f1_score(metrics_test_labels, predictions, average = 'micro'))
            accuracy_scores_90.append(classifier_split_test.score(test_data_split, test_labels_split))
            
#all f1 scores were the same regardless of random state and train test split


# In[22]:


#calculating the means and standard deviations for the f1 scores.
import statistics

avg_f1_score_50 = statistics.mean(f1_scores_50)
avg_f1_score_60 = statistics.mean(f1_scores_60)
avg_f1_score_70 = statistics.mean(f1_scores_70)
avg_f1_score_80 = statistics.mean(f1_scores_80)
avg_f1_score_90 = statistics.mean(f1_scores_90)

print(avg_f1_score_50)

std_dev_f1_50 = statistics.stdev(f1_scores_50, avg_f1_score_50)
std_dev_f1_60 = statistics.stdev(f1_scores_60, avg_f1_score_60)
std_dev_f1_70 = statistics.stdev(f1_scores_70, avg_f1_score_70)
std_dev_f1_80 = statistics.stdev(f1_scores_80, avg_f1_score_80)
std_dev_f1_90 = statistics.stdev(f1_scores_90, avg_f1_score_90)

print(std_dev_f1_50)


# In[23]:


#calculating mean and standard deviation for accuracy

avg_accuracy_50 = statistics.mean(accuracy_scores_50)
avg_accuracy_60 = statistics.mean(accuracy_scores_60)
avg_accuracy_70 = statistics.mean(accuracy_scores_70)
avg_accuracy_80 = statistics.mean(accuracy_scores_80)
avg_accuracy_90 = statistics.mean(accuracy_scores_90)

print(avg_accuracy_50)

std_dev_accuracy_50 = statistics.stdev(accuracy_scores_50, avg_accuracy_50)
std_dev_accuracy_60 = statistics.stdev(accuracy_scores_60, avg_accuracy_60)
std_dev_accuracy_70 = statistics.stdev(accuracy_scores_70, avg_accuracy_70)
std_dev_accuracy_80 = statistics.stdev(accuracy_scores_80, avg_accuracy_80)
std_dev_accuracy_90 = statistics.stdev(accuracy_scores_90, avg_accuracy_90)

print(std_dev_accuracy_50)


# In[24]:


#graphically displaying means and standard deviations for f1_score.
import matplotlib as mpl

mpl.use('agg')

plot_f1 = [f1_scores_50,f1_scores_60,f1_scores_70,f1_scores_80,f1_scores_90]

plt_1, ax1 = plt.subplots()
ax1.set_title('F1 Score vs Proportion of data used for training and different random states')
ax1.set_ylabel('F1 Score')
ax1.set_xlabel('Proportion of data used for training')
ax1.set_xticklabels(['50%', '60%', '70%', '80%', '90%'])
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.boxplot(plot_f1)


plt_1.savefig('f1_boxplot.png', bbox_inches='tight')


# In[25]:


#graphically displaying means and standard deviations for f1_score.

plot_acc = [accuracy_scores_50,accuracy_scores_60,accuracy_scores_70,accuracy_scores_80,accuracy_scores_90]

plt_2, ax2 = plt.subplots()
ax2.set_title('Classification Accuracy vs Proportion of data used for training and different random states')
ax2.set_ylabel('Classification Accuracy')
ax2.set_xlabel('Proportion of data used for training')
ax2.set_xticklabels(['50%', '60%', '70%', '80%', '90%'])
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.boxplot(plot_acc)


plt_2.savefig('accuracy_boxplot.png', bbox_inches='tight')


# In[26]:


#building a "light" model (rbf kernel function) - training and testing the model
print(len(light_data.columns))
light_classifier = SVC()
light_train_start = time.time()
light_classifier.fit(light_train_data, light_train_labels)
light_train_time = time.time() - light_train_start
print("training time:", light_train_time, "seconds")

light_test_start = time.time()
print(light_classifier.score(light_test_data, light_test_labels))
light_test_time = time.time() - light_test_start
print("test duration:", light_test_time, "seconds")


# In[27]:


#building a "light" model (rbf kernel function) - testing the models f1 score
filler_train, light_metrics_test_data, filler_labels, light_metrics_test_labels = train_test_split(light_data, light_data_labels, train_size = 0.5, random_state = 46)
#getting these metrics for the classifer (rbf)
start_light_predict = time.time()
predicted_labels_light = light_classifier.predict(light_metrics_test_data)
run_time_light_predict = time.time() - start_light_predict
print("Prediction time:", run_time_light_predict, "seconds")

f1_classifier_light = f1_score(light_metrics_test_labels, predicted_labels_light, average = 'micro')
precision_classifier_light = precision_score(light_metrics_test_labels, predicted_labels_light, average = 'micro')
recall_classifier_light = recall_score(light_metrics_test_labels, predicted_labels_light, average = 'micro')

print("F1 score for light classifier (rbf kernel):", f1_classifier_light)
print("Precision score for light classifier (rbf kernel):", precision_classifier_light)
print("Recall score for light classifier (rbf kernel):", recall_classifier_light)


# In[28]:


#building a "light" model (poly kernel function) - training and testing the model
light_classifier_poly = SVC(kernel = 'poly')
light_train_start_poly = time.time()
light_classifier_poly.fit(light_train_data, light_train_labels)
light_train_time_poly = time.time() - light_train_start_poly
print("training time:", light_train_time_poly)

light_test_start_poly = time.time()
print(light_classifier_poly.score(light_test_data, light_test_labels))
light_test_time_poly = time.time() - light_test_start_poly
print("test duration:", light_test_time_poly)


# In[29]:


#building a "light" model (poly kernel function) - testing the models f1 score
#getting these metrics for the classifer (poly)
start_light_predict_poly = time.time()
predicted_labels_light_poly = light_classifier_poly.predict(light_metrics_test_data)
run_time_light_predict_poly = time.time() - start_light_predict_poly
print("Prediction time:", run_time_light_predict_poly, "seconds")

f1_classifier_light_poly = f1_score(light_metrics_test_labels, predicted_labels_light_poly, average = 'micro')
precision_classifier_light_poly = precision_score(light_metrics_test_labels, predicted_labels_light_poly, average = 'micro')
recall_classifier_light_poly = recall_score(light_metrics_test_labels, predicted_labels_light_poly, average = 'micro')

print("F1 score for light classifier (poly kernel):", f1_classifier_light_poly)
print("Precision score for light classifier (poly kernel):", precision_classifier_light_poly)
print("Recall score for light classifier (poly kernel):", recall_classifier_light_poly)


# In[30]:


from sklearn.model_selection import KFold
#implementing K cross-fold validation as a method of splitting data for training and validation.
#utilising the light data as it provides comparible performance and reduces training and prediction times.

kf = KFold(n_splits = 5, random_state = 62, shuffle = True)
print(kf)

kfold_data = light_data.copy()
kfold_labels = light_data_labels.copy()
kfold_data = kfold_data.to_numpy()
kfold_labels  = kfold_labels.to_numpy()
#testing effectiveness with kfold cross validation - rbf kernel
kfold_classifier = SVC()

kfold_f1score = []
print("Using k-fold cross validation:")

for train_index, test_index in kf.split(kfold_data):
     data_train, data_test = kfold_data[train_index], kfold_data[test_index]
     labels_train, labels_test = kfold_labels[train_index], kfold_labels[test_index]
     kfold_classifier.fit(data_train, labels_train)
     accuracy_score = kfold_classifier.score(data_test, labels_test)
     print("Accuracy:", accuracy_score)
     kfold_f1score.append(f1_score(labels_test, kfold_classifier.predict(data_test), average = 'micro'))
     print("F1 Score:", f1_score(labels_test, kfold_classifier.predict(data_test), average = 'micro'))


# In[31]:


#calculating mean F1 score and standard deviation for KFold classifier - rbf kernel
mean_f1_kfold = statistics.mean(kfold_f1score)

stddev_f1_kfold = statistics.stdev(kfold_f1score, mean_f1_kfold)

print("Mean F1 Score - rbf kernel:", mean_f1_kfold)
print("Standard deviation in F1 score - rbf kernel:", stddev_f1_kfold)


# In[32]:


#using different kernel functions to test effectiveness - poly kernel
kfold_classifier_poly = SVC(kernel = 'poly')

kfold_f1score_poly = []
print("Using k-fold cross validation:")

for train_index, test_index in kf.split(kfold_data):
     data_train, data_test = kfold_data[train_index], kfold_data[test_index]
     labels_train, labels_test = kfold_labels[train_index], kfold_labels[test_index]
     kfold_classifier_poly.fit(data_train, labels_train)
     accuracy_score = kfold_classifier_poly.score(data_test, labels_test)
     print("Accuracy - poly kernel:", accuracy_score)
     kfold_f1score_poly.append(f1_score(labels_test, kfold_classifier_poly.predict(data_test), average = 'micro'))
     print("F1 Score - poly kernel:", f1_score(labels_test, kfold_classifier_poly.predict(data_test), average = 'micro'))


# In[40]:


print(data.columns)


# In[33]:


#calculating mean F1 score and standard deviation for KFold classifier - poly kernel
mean_f1_kfold_poly = statistics.mean(kfold_f1score_poly)

stddev_f1_kfold_poly = statistics.stdev(kfold_f1score_poly, mean_f1_kfold_poly)

print("Mean F1 Score - poly kernel:", mean_f1_kfold_poly)
print("Standard deviation in F1 score - poly kernel:", stddev_f1_kfold_poly)


# In[ ]:


#using different kernel functions to test effectiveness - linear kernel
kfold_classifier_linear = SVC(kernel = 'linear')

kfold_f1score_linear = []
print("Using k-fold cross validation:")

for train_index, test_index in kf.split(kfold_data):
     data_train, data_test = kfold_data[train_index], kfold_data[test_index]
     labels_train, labels_test = kfold_labels[train_index], kfold_labels[test_index]
     kfold_classifier_linear.fit(data_train, labels_train)
     accuracy_score = kfold_classifier_linear.score(data_test, labels_test)
     print("Accuracy - linear kernel:", accuracy_score)
     kfold_f1score_linear.append(f1_score(labels_test, kfold_classifier_linear.predict(data_test), average = 'micro'))
     print("F1 Score - linear kernel:", f1_score(labels_test, kfold_classifier_linear.predict(data_test), average = 'micro'))


# In[34]:


#calculating mean F1 score and standard deviation for KFold classifier - linear kernel
mean_f1_kfold_linear = statistics.mean(kfold_f1score_linear)

stddev_f1_kfold_linear = statistics.stdev(kfold_f1score_linear, mean_f1_kfold_linear)

print("Mean F1 Score - linear kernel:", mean_f1_kfold_linear)
print("Standard deviation in F1 score - linear kernel:", stddev_f1_kfold_linear)


# In[34]:


#formatting training data and lables
training_data = training_data.to_numpy()
labels = labels.to_numpy()

#Performing K-fold cross validation on K-fold cross validation on Classifier 1- rbf kernel
#Using "full" training data

kfold_f1score_rbffull = []
print("Using k-fold cross validation:")

for train_index, test_index in kf.split(training_data):
     data_train, data_test = training_data[train_index], training_data[test_index]
     labels_train, labels_test = labels[train_index], labels[test_index]
     classifier_1.fit(data_train, labels_train)
     accuracy_score = classifier_1.score(data_test, labels_test)
     print("Accuracy - linear kernel:", accuracy_score)
     kfold_f1score_rbffull.append(f1_score(labels_test, classifier_1.predict(data_test), average = 'micro'))
     print("F1 Score - linear kernel:", f1_score(labels_test, classifier_1.predict(data_test), average = 'micro'))


# In[35]:


#Performing K-fold cross validation on K-fold cross validation on Classifier 2- poly kernel
#Using "full" training data

kfold_f1score_polyfull = []
print("Using k-fold cross validation:")

for train_index, test_index in kf.split(training_data):
     data_train, data_test = training_data[train_index], training_data[test_index]
     labels_train, labels_test = labels[train_index], labels[test_index]
     classifier_2.fit(data_train, labels_train)
     accuracy_score = classifier_2.score(data_test, labels_test)
     print("Accuracy - linear kernel:", accuracy_score)
     kfold_f1score_polyfull.append(f1_score(labels_test, classifier_2.predict(data_test), average = 'micro'))
     print("F1 Score - linear kernel:", f1_score(labels_test, classifier_2.predict(data_test), average = 'micro'))


# In[ ]:


#Performing K-fold cross validation on K-fold cross validation on Classifier 3- linear kernel
#Using "full" training data

kfold_f1score_linearfull = []
print("Using k-fold cross validation:")
for train_index, test_index in kf.split(training_data):
     data_train, data_test = training_data[train_index], training_data[test_index]
     labels_train, labels_test = labels[train_index], labels[test_index]
     classifier_3.fit(data_train, labels_train)
     accuracy_score = classifier_3.score(data_test, labels_test)
     print("Accuracy - linear kernel:", accuracy_score)
     kfold_f1score_linearfull.append(f1_score(labels_test, classifier_3.predict(data_test), average = 'micro'))
     print("F1 Score - linear kernel:", f1_score(labels_test, classifier_3.predict(data_test), average = 'micro'))


# In[39]:


#graphically displaying means and standard deviations for each of the classifiers tested using K-fold cross validation

K_fold_F1 = [kfold_f1score_rbffull,kfold_f1score,kfold_f1score_polyfull,kfold_f1score_poly]

plt_3, ax3 = plt.subplots()
ax3.set_title('F1 Score vs Dataset and kernel function used')
ax3.set_ylabel('F1 scores')
ax3.set_xlabel('Dataset and kernel function')
ax3.set_xticklabels(['full, rbf', 'light, rbf', 'full, poly', 'light, poly'])
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()
ax3.boxplot(K_fold_F1)


plt_3.savefig('F1_Scores_Kfold.png', bbox_inches='tight')


# In[83]:


#VISUALISATION - KMeans clustering on the dataset, 4 clusters will be used to begin with (one for each class)
#as it is important to see whether there can be common features discovered between the 
#data in each cluster.
#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters = 2)

#kmeans_data = training_data.copy()
#print(kmeans_data.columns)
#kmeans_data = kmeans_data.to_numpy()
#train the model
#kmeans.fit(kmeans_data)
#y_kmeans = kmeans.predict(kmeans_data)


# In[86]:


#plotting the KMeans data
#plt.title("Loan amount vs. FEATURE")
#plt.ylabel("Loan amount")
#plt.xlabel("FEATURE")
#THESE NEED TO BE CHANGED TO BE ACTUALLY VALUABLE
#plotting data
#plt.scatter(kmeans_data[:,1],kmeans_data[:,0], c=y_kmeans, s=50, cmap = 'viridis')
#plot the centroids on the graph
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:,1],centroids[:,0], c='black', s=200, alpha=0.5)
#plt.savefig('test1.png', bbox_inches='tight')

