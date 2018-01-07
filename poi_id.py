#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 
                      'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock',
                      'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']
# You will need to use more features
features_list = poi_label + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#Some information on the data
print("Total number of data points: " + str(len(data_dict)))

#Total number of POI adapted from - explore_enron_data.py
count_poi = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
       count_poi += 1
print("Total number of poi: " + str(count_poi))
print("Total number of non-poi: " + str(len(data_dict) - count_poi))


### Task 2: Remove outliers

### Function adapted from Outlier Lesson - enron_outliers.py
import matplotlib.pyplot
def enron_outlier(data_set, feature_x, feature_y):
    """
    This function is from outlier lesson modified to take a dict
    and two features to compare to
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()



print(enron_outlier(data_dict, 'total_payments', 'total_stock_value'))
print(enron_outlier(data_dict, 'exercised_stock_options', 'total_stock_value'))
print(enron_outlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))

#At this point I decided plot one more and remove TOTAL outlier since it wasn't giving us
#any insights due to the TOTAL being added from the data.
print(enron_outlier(data_dict, 'salary', 'bonus'))
data_dict.pop("TOTAL", 0)

print(enron_outlier(data_dict, 'total_payments', 'other'))

#Looking for NaN values in Financial feature by modifying it from data set lesson into a dict
financial_feature_nan = {}
for person in data_dict:
    financial_feature_nan[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            financial_feature_nan[person] += 1
sorted(financial_feature_nan.items(), key=lambda x: x[1])

#Looking for NaN values in Email feature
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Creating total amount receipt shared with poi to make a new feature
total_receipt = 0
count_people_receipt = 0
for person_receipt in my_dataset:
        current_receipt = my_dataset[person_receipt]['shared_receipt_with_poi']
        #print(my_dataset[person_receipt].values())
        if current_receipt != "NaN":
            print(current_receipt)
            total_receipt += my_dataset[person_receipt]['shared_receipt_with_poi']
            count_people_receipt += 1
print('Total amount of receipt: ' + str(total_receipt))
print('People with receipt with poi: ' + str(count_people_receipt))

for person in my_dataset:
    receipt_shared = my_dataset[person]['shared_receipt_with_poi']
    if receipt_shared != "NaN":
        my_dataset[person]['shared_receipt_percent'] = receipt_shared/float(total_receipt)
    else:
        my_dataset[person]['shared_receipt_percent'] = 0
    person_bonus = my_dataset[person]['bonus']
    person_salary = my_dataset[person]['salary']
    if person_bonus != "NaN" and person_salary != "NaN":
        my_dataset[person]['bonus_salary_ratio'] = person_bonus/float(person_salary)
    else:
        my_dataset[person]['bonus_salary_ratio'] = 0
    from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['from_poi_ratio'] = from_poi/float(to_msg)
    else:
        my_dataset[person]['from_poi_ratio'] = 0
    to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['to_poi_ratio'] = to_poi/float(from_msg)
    else:
        my_dataset[person]['to_poi_ratio'] = 0
        
features_list.append('shared_receipt_percent')
features_list.append('bonus_salary_ratio')
features_list.append('to_poi_ratio')
features_list.append('from_poi_ratio')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Sorted features:")
scores = zip(features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
K_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k] 
print("New Features:")
print(K_features_list)

updated_features_list = K_features_list + ['shared_receipt_percent', 'from_poi_ratio']
print("Updated Features:")
print(updated_features_list)


data = featureFormat(my_dataset, updated_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#from time import time
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score
#t0 = time()
#clf = GaussianNB()
#clf.fit(features_train, labels_train)

#print "\ntraining time:", round(time()-t0, 3), "s"

#predict
#t0 = time()

#pred = clf.predict(features_test)
#print "predicting timer:", round(time()-t0, 3), "s"
#accuracy = accuracy_score(pred, labels_test)

#print accuracy

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

nb = GaussianNB()
dtc = DecisionTreeClassifier()
svc = SVC()
knc = KNeighborsClassifier()

#Adapted from percision and recall lesson and split cross validation in scikit-learn
def evaluate_clf(grid_search, features, labels, params, iters=100):
    accuracy = []
    precision = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        accuracy = accuracy + [accuracy_score(labels_test, predictions)] 
        precision = precision + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(accuracy))
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    # Pick the classifier with the best tuned parameters
    best_params = grid_search.best_estimator_.get_params()
    #print "\n", "Best parameters are: ", best_params, "\n"
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))    

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Naive's Bayes test and tuning
nb_parameters = {}
nb_gridsearch = GridSearchCV(nb, nb_parameters)
print("Evaluation and tuning  for naive bayes model")
evaluate_clf(nb_gridsearch, features, labels, nb_parameters)
#accuracy: 0.861428571429
#precision: 0.457826839827
#recall:    0.377984848485

# Example starting point. Try investigating other evaluation techniques!
#Decision Tree Classifier tests and tuning.
#The following takes a while to run
dtc_parameters = { 'criterion':['gini', 'entropy'],
                  'splitter':['best', 'random'],
                  'max_depth':[None, 1, 2, 3, 4],
                  'min_samples_split':[1.0, 2, 3, 4, 25],
                  'min_samples_leaf':[1, 2, 3, 4],
                  'min_weight_fraction_leaf':[0, 0.25, 0.5],
                  'class_weight':[None, 'balanced'],
                  'random_state':[None, 42]
                  }

dtc_gridsearch = GridSearchCV(dtc, dtc_parameters)
#print("Evaluation and tuning for Decision Tree Classifier")
#evaluate_clf(dtc_gridsearch, features, labels, dtc_parameters)
#accuracy: 0.819047619048
#precision: 0.306673604174
#recall:    0.299607864358
#Best parameters are:  {'presort': False, 'splitter': 'random', 
#                       'min_impurity_decrease': 0.0, 'max_leaf_nodes': None, 
#                       'min_samples_leaf': 1, 'min_samples_split': 4, 
#                       'min_weight_fraction_leaf': 0, 'criterion': 'gini', 
#                       'random_state': None, 'min_impurity_split': None, 
#                       'max_features': None, 'max_depth': 1, 'class_weight': None}


#SVC tests and tunning
#Warning takes extremely long to run
svc_parameters = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}  

svc_gridsearch = GridSearchCV(svc, svc_parameters)
#print("Evaluation and tuning for svm model")
#evaluate_clf(svc_gridsearch, features, labels, svc_parameters)
#accuracy: 0.866428571429
#precision: 0.141666666667
#recall:    0.0384523809524
#Best parameters are:  {'C': 1.0, 'kernel': 'linear', 'degree': 3, gamma: 'auto',
#                       'coef0': 0.0, 'shrinking': True, 'probability': False,
#                       'tol': 0.001, 'cache_size': 200, 'class_weight': None, 
#                       'verbose': False, 'max_iter': -1, 
#                       'decision_function_shape': 'ovr', 'random_state': None}



#Kneighbors test and tunning
knc_parameters = {'n_neighbors':[1, 2, 3, 4, 5],
                   'leaf_size':[1, 10, 30, 60],
                   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                   }

knc_gridsearch = GridSearchCV(knc, knc_parameters)
#print("Evaluation and tuning for kneighbors")
#evaluate_clf(knc_gridsearch, features, labels, knc_parameters)
#accuracy: 0.865238095238
#precision: 0.325833333333
#recall:    0.152738095238
#Best parameters are:  {'n_neighbors': 4, 'n_jobs': 1, 'algorithm': 'auto', 
#                       'metric': 'minkowski', 'metric_params': None, 'p': 2, 
#                       'weights': 'uniform', 'leaf_size': 1}


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()

dump_classifier_and_data(clf, my_dataset, updated_features_list)