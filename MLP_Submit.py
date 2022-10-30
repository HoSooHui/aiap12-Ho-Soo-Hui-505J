#====================================================================================================================
#
# Program ID :  MLP_submit.py
# Description:  FOR AIAP ASSESSMENT 2
#               TASK 2 - End-to-end Machine Learning Pipeline 
# Process    :  Explore Data  --> Fit Model  --> Evaluate Model  --> Deploy Model            
#               
# OUTLINE    :  This program is structured as per below:
#               (1) Common Functions
#               (2) Initialisation
#                   (i) Import Libraries, Initial Setup
#                   (ii) Read in Data 
#               (3) Data Exploration and Transformation
#               (4) Data Preparation (Preprocessing)
#
#               (5) KNN Model 1 - Without numeric scaling, and comprises of both categorical and numerical predictors
#               (6) KNN Model 2 - With numeric scaling, and comprises of both categorical and numerical predictors
#               (7) KNN Model 3 - With numeric scaling, and comprises of numerical predictors (aka no categorical variable)
#
#               (8) Decision Tree Model 4 - Gini, and comprises of both categorical and numerical predictors
#               (9) Decision Tree Model 5 - Gini, and comprises numerical predictors (aka no categorical variable)            
#
#               (10) Summary and Interpretation
#
#====================================================================================================================
# (1) List of Common Functions
#====================================================================================================================

# Function to recode the "Attrition" response variable
# Called by: (3) Data Exploration and Transformation
# Calling  : NA
def recode_attrition(attrition_value):
    if attrition_value == 0:
        attrition_value = 'No Attrition'
    else:
        attrition_value = 'Attrition'
    return attrition_value

# Common function to set "-1" to missing value
# Called by: (3) Data Exploration and Transformation
# Calling  : NA
def set_null (na_value):
    if na_value == -1 :
        na_value = None
 
    return na_value

# Function to recode Qualification Predictors
# Called by: (3) Data Exploration and Transformation
# Calling  : NA
def recode_qualification(qual_value):
    if qual_value == "Bachelor's":
        qual_value = 'Bachelor'
    elif qual_value == "Master's":
        qual_value= 'Master'
    return qual_value

# Function to split training and testing data set, containing both categorical and numerical predictors in same return file
# Called by: KNN and Deision Tree Modeling
# Calling  : NA
def split_train_test_set (x, y):
    
    # random_state used so as to have the same result for re-run and comparison across all models 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=12345)

    print ( f'No. of records in training set : {len(x_train)} ')
    print ( f'No. of records in testing set  : {len(x_test)} \n')

    return x_train, x_test, y_train, y_test

# Function to split training and testing data set, separating categorical and numerical predictors
# Called by: KNN and Deision Tree Modeling
# Calling  : NA
def split_train_test_num_cat_set (x_num, x_cat, y):
    
    # random_state used so as to have the same result for re-run and comparison across all models 
    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(x_num, x_cat, y, test_size=0.20,random_state=12345)

    print ( f'No. of records in NUM training set : {len(x_num_train)} ')
    print ( f'No. of records in NUM testing set  : {len(x_num_test)} \n')
    print ( f'No. of records in CAT training set : {len(x_cat_train)} ')
    print ( f'No. of records in CAT testing set  : {len(x_cat_test)} \n')

    return x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test

# Function to print evaluation matrix for evaluation
# Called by: KNN and Decision tree Modeling
# Calling  : NA
def print_eval_matrix(y_test, y_pred, title ="Others"):

    print (f'{title}')

    print ('Confusion Matrix : ')
    print (confusion_matrix(y_test, y_pred))
    print (' ')

    print ('Classification Report : ')
    print (classification_report(y_test, y_pred))
    print (' ')

    return

#===================================================================================================================
# (2) Initialisation
#===================================================================================================================

# 2(i) Import Libraries
# ----------------------------
# from typing import Concatenate
import numpy as np 
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Disable warning, after verification and all in order
import warnings
# warnings.filterwarnings("ignore")

# 2(ii) Read in Data
# ----------------------------
# Attempt to connect to the SQLite attrition database
try:
    conn = sqlite3.connect("data/attrition.db")  

# Print the exception for followup, if the connection attempt is unsuccessful
except Exception as err:
    print(err)
      
# For successful connection, read the attrition into data frame for subsequent data exploration.
cursor = conn.cursor()
df_raw = pd.read_sql_query('SELECT * FROM attrition', conn)

# Close db connection
conn.close()

#===================================================================================================================
# (3) Data Exploration and Transformation
#===================================================================================================================
#  3 (i) List of data preparation activities identified during Task 2 Exploratory Data Analysis are:
#      - Recode column "Attrition" response variable
#      - Standardise the data in "Travel Time" column to use common unit in mins
#      - Replace "Age" with "-1" (or interpreted as missing values), with the mean of Age
#      - Replace "Birth Year" with "-1" (or interpreted as missing values), with the mode of Birth Year
#      - Drop the unmeaningful "Member Unique ID" and Inconsistent "Travel Time" columns
#      - Create a new interaction column "Usage Hours" to store the total hours spent in the club per week
#      - For the "Qualification" predictor, "Bachelor" and "Bachelor's" coding to be aligned as one value, as well as the 
#        "Master" and "Master's" coding is to be aligned too
#      Note that the above mentioned activities were carried out and verified in the earlier EDA process in ipynb.
#
# Apply the "Attrition" recode function
df_raw['Attrition']= df_raw['Attrition'].apply(recode_attrition) 

# Create a new column "Travel Time (mins)" to store the standardised the Travel Time in mins. 
# Move the houred "Travel Time" to the new column. Convert to min by multiplying 60.
df_raw['Travel Time (mins)'] = df_raw['Travel Time'].str.replace(' hours','') [ (df_raw['Travel Time'].str.contains('hours') ) ]
df_raw['Travel Time (mins)'] = df_raw['Travel Time (mins)'].astype(float) * 60
# Move the remaining minuted "Travel Time" to the new column
df_raw['Travel Time (mins)'].fillna( df_raw['Travel Time'].str.replace(' mins',''), inplace = True)
df_raw['Travel Time (mins)']=df_raw['Travel Time (mins)'].astype(float)

# Apply the set_null function for "Age" with "-1"
df_raw['Age']= df_raw['Age'].apply(set_null) 
# Fill the missing value with mean of age
df_raw['Age']=df_raw['Age'].fillna(df_raw['Age'].mean())

# Apply the set_null function for "Birth Year" with "-1"
df_raw['Birth Year']= df_raw['Birth Year'].apply(set_null) 
# Fill the missing value with mode for birth year
df_raw['Birth Year']=df_raw['Birth Year'].fillna(df_raw['Birth Year'].mode() [0])

# Drop the unmeaningful "Member Unique ID" and Inconsistent "Travel Time" columns
# The inconsistemt "Travel Time" column was replaced by the new column "Travel Time (mins)"
df_clean = df_raw.drop(['Member Unique ID','Travel Time'], axis=1)

# Create a new column "Usage Hours" to store the total hours spent in the club per week
df_clean['Usage Hours'] = df_clean['Usage Rate'] * df_clean['Usage Time'] 
df_clean['Qualification']= df_clean['Qualification'].apply(recode_qualification) 

# df_clean.to_csv("data/attrition_cleaned1.csv",index=False)

#===================================================================================================================
# (4) Prepare Train and Test Data Set for splitting (Pre-processing)
#===================================================================================================================

# Select predictor and response data set
x = df_clean.drop(['Attrition'],axis=1)
y = df_clean['Attrition'] [df_clean['Attrition'] != '']

# Select and extract categorical cols for predictor data set
categorical_cols = df_clean.select_dtypes(include='object').columns
x_cat = df_clean[categorical_cols]
x_cat = x_cat.drop(['Attrition'],axis=1)

# Select and extract numerical cols for predictor data set
numerical_cols = df_clean.select_dtypes(include='number').columns
x_num = df_clean[numerical_cols]


#===================================================================================================================
# (5) KNN Model 1 - Without numeric scaling, and comprises of both categorical and numerical predictors
#===================================================================================================================

print ('Starting KNN modeling and Fitting - Model 1')
print ('Without numeric scaling, and comprises of both categorical and numerical predictors')
print ('=================================================================================== \n')

# Get train and test data set
x_train, x_test, y_train, y_test = split_train_test_set (x, y)

# Apply the category encoder to the train and test sets
l_encoder = LabelEncoder()
for col_name in x_train.columns:
    if x_train[col_name].dtype == object:
        x_train[col_name] = l_encoder.fit_transform(x_train[col_name])
    else:
        pass

for col_name in x_test.columns:
    if x_test[col_name].dtype == object:
        x_test[col_name] = l_encoder.fit_transform(x_test[col_name])
    else:
        pass

# KNN model training, with n=9 obtained from the graph generated below, for lowest k with relatively low errors
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(x_train, y_train)

# KNN model testing
y_pred = classifier.predict(x_test)

print_eval_matrix(y_test, y_pred,'KNN Model 1')

# Calculating error for K values between 1 and 20
# ------------------------------------------------
knn_error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    knn_error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), knn_error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('KNN Model 1 - Error Rate and K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error') 
plt.show()


#===================================================================================================================
#  (6) KNN Model 2 - With numeric scaling, and comprises of both categorical and numerical predictors
#===================================================================================================================

print ('Starting KNN modeling and Fitting - Model 2')
print ('With numeric scaling, and comprises of both categorical and numerical predictors')
print ('================================================================================ \n')

# Get train and test data set
x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test= split_train_test_num_cat_set (x_num, x_cat, y)

# scaling of numerical predictors
scaler = StandardScaler()
scaler.fit(x_num_train)

x_num_scaled_train = pd.DataFrame(scaler.transform(x_num_train))
x_num_scaled_test = pd.DataFrame(scaler.transform(x_num_test))

# Scaler removes the column names, so put back the columns
x_num_scaled_train.columns = x_num.columns
x_num_scaled_test.columns = x_num.columns

#Apply the categorical label encoder 
l_encoder = LabelEncoder()
for cat_col in x_cat_train.columns:
    x_cat_train[cat_col] = l_encoder.fit_transform(x_cat_train[cat_col])

for cat_col in x_cat_test.columns:
    x_cat_test[cat_col] = l_encoder.fit_transform(x_cat_test[cat_col])

# concatenate both the categorical and numerical predictors
x_full_train = np.concatenate((x_num_scaled_train,x_cat_train), axis=1)
x_full_test = np.concatenate((x_num_scaled_test,x_cat_test), axis=1)

# KNN model training, with n=9 obtained from the graph generated below, for lowest k with relatively low errors
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(x_full_train, y_train)

# KNN model testing
y_pred = classifier.predict(x_full_test)

print_eval_matrix(y_test, y_pred,'KNN Model 2')

# Calculating error for K values between 1 and 20
# ------------------------------------------------
knn_error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_full_train, y_train)
    pred_i = knn.predict(x_full_test)
    knn_error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), knn_error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('KNN Model 2 - Error Rate and K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error') 
plt.show()


#===================================================================================================================
# (7) KNN Model 3 - With numeric scaling, and comprises of numerical predictors (aka no categorical variable)
#===================================================================================================================
#
print ('Starting KNN modeling and Fitting - model 3')
print ('With numeric scaling, and comprises of only numerical predictors')
print ('================================================================ \n')

# Get train and test data set
x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test= split_train_test_num_cat_set (x_num, x_cat, y)

# scaling of numerical predictors
scaler = StandardScaler()
scaler.fit(x_num_train)

x_num_scaled_train = pd.DataFrame(scaler.transform(x_num_train))
x_num_scaled_test = pd.DataFrame(scaler.transform(x_num_test))

# Scaler removes the column names, so put back the columns
x_num_scaled_train.columns = x_num.columns
x_num_scaled_test.columns = x_num.columns

# KNN model training, with n=8 obtained from the graph generated below, for lowest k with relatively low errors
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_num_scaled_train, y_train)

#KNN testing
y_pred = classifier.predict(x_num_scaled_test)

print_eval_matrix(y_test, y_pred,'KNN model 3')

# Calculating error for K values between 1 and 20
# ------------------------------------------------
knn_error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_num_scaled_train, y_train)
    pred_i = knn.predict(x_num_scaled_test)
    knn_error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), knn_error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('KNN model 3 : Error Rate and K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error') 
plt.show()

#===================================================================================================================
# (8) Decision Tree Model 4 - Gini, and comprises of both categorical and numerical predictors
#===================================================================================================================

print ('Starting decision tree model 4')
print ('Gini, and comprises of both categorical and numerical predictors')
print ('================================================================\n')

x_train, x_test, y_train, y_test = split_train_test_set (x, y)

# Apply the encoder to the train and test sets
l_encoder = LabelEncoder()
for col_name in x_train.columns:
    if x_train[col_name].dtype == object:
        x_train[col_name] = l_encoder.fit_transform(x_train[col_name])
    else:
        pass

for col_name in x_test.columns:
    if x_test[col_name].dtype == object:
        x_test[col_name] = l_encoder.fit_transform(x_test[col_name])
    else:
        pass

clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth = 4, min_samples_leaf = 6)
clf_gini.fit(x_train, y_train)
  
y_pred = clf_gini.predict(x_test)
print_eval_matrix(y_test, y_pred,'Decision tree model 4')
      
# plot the tree
# --------------
features = x_train.columns
plt.figure(figsize=(10,6))
tree.plot_tree(clf_gini, feature_names=features, fontsize=7)
plt.title ('Decision tree model 4')
plt.show()


#===================================================================================================================
# (9) Decision Tree Model 5 - Gini, and comprises numerical predictors (aka no categorical variable)   
#===================================================================================================================

print ('Starting decision tree model 5')
print ('Gini, and comprises of only numerical predictors')
print ('================================================\n')

# Get train and test data set
x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test= split_train_test_num_cat_set (x_num, x_cat, y)

clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth=4, min_samples_leaf=6)
clf_gini.fit(x_num_train, y_train)  
    
y_pred = clf_gini.predict(x_num_test)
print_eval_matrix(y_test, y_pred,'Decision tree model 5')
      

# Plot the tree
# -------------
features = x_num.columns
plt.figure(figsize=(10,6))
tree.plot_tree(clf_gini, feature_names=features,fontsize=7)
plt.title ('Decision tree model 5')
plt.show()


#====================================================================================================================
# (10) Summary and Interpretation
#====================================================================================================================

print ( f'Summary:')
print ( f'========')
print ( f'(i) KNN classifier')
print ( f'    - Both KNN model 1, 2 have accuracy of 0.81 to 0.82.')
print ( f'    - KNN model 2 with scaling of numeric predictors performed a slight better (at ~0.01) than the KNN model 1 with NO scaling.')
print ( f'    - Recall rating for Attrition is <= 0.06 for both models (with KNN model 2 a slight better in predicting Attrition).')
print ( f'      Thus, both models most probably NOT able to predict the Attrition correctly, even though with high accuracy.\n')
print ( f'(ii) Decision Tree')
print ( f'     - Decision Tree Model 4 has high accuracy of 0.82, but low Recall rating for Attrition (at ~0.04) as well. The high accuracy is')
print ( f'       from the high performance in predicting No Attrition.')
print ( f'     - Through the tree plot which shows the more significant predictors from the top were numerical predictors, such as usage')
print ( f'       hours, months, etc.')
print ( f'     - Thus, decision tree model 5 and KNN model 3 were fitted to find out the significant of numerical predictors. \n')
print ( f'In short:')
print ( f'(i) Inbalance proportion of Atrition (17%) vs No Attrition (83%) might have resulted in lower performance for predicting Attrition.')
print ( f'(ii) The significant predictors showed at the top of the decision tree plot were numerical predictors.\n')
print ( f'Thanks for reading. \n')
print ( f'<---------------------- End --------------------->')


