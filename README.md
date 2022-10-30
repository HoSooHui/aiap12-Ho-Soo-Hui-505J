# aiap12-Ho-Soo-Hui-505J
AIAP Assessment

Full Name: Ho Soo Hui

Email address (stated in the Application Form) : Soohui-ho@hotmail.com

In the 'Main' branch, there are the below commited files:

(1) eda.ipynb. 
    This is on Task 1 - Exploratory Data Analysis.
    Below is the outline written in the eda.ipynb, step 1:
    
    # OUTLINE    :  THIS PROGRAM CONTAINS THE FOLLOWING KEY PROCESSES
    #               (1) IMPORT LIBRARIES AND INITIALISATION
    #               (2) READ IN DATA AND VERIFICATION
    #               (3) ESSENTIAL DATA CLEANING AND TRANSFORMATION
    #
    #               (4) EXPLORE QUALITATIVE CATEGORICAL PREDICTORS AND RESPONSE VARIABLES (Pie charts)
    #               (5) EXPLORE QUANTITATIVE CONTINUOUS AND DISCRETE PREDICTORS (Corelation, Heatmap and Pairplots)
    #               (6) EXPLORE PREDICTORS AND RESPONSE (Box plots)
    #
    #               (7) CONCLUSION AND FOLLOWUP TO TASK 2
    
    For each of the 7 processes, the summary/observation could be found at the end of the process, and before the begining of the next process.
    
    In addition, the purpose of each step, and the finding for earlier step, are also explained within the step as comments "#".
    
    Below is conclusion at the last step 42:
    
    # (7) CONCLUSION AND FOLLOWUP TO TASK 2
    # ===================================================================================================================
    #
    #  (i) List of data preparation activities recommended for Task 2 includes:
    #      - Recode column "Attrition" response variable
    #      - Standardise the data in "Travel Time" column to use common unit in mins
    #      - Replace "Age" with "-1" (or interpreted as missing values), with the mean of Age
    #      - Replace "Birth Year" with "-1" (or interpreted as missing values), with the mode of Birth Year
    #      - Drop the unmeaningful "Member Unique ID" and Inconsistent "Travel Time" columns
    #      - Create a new interaction column "Usage Hours" to store the total hours spent in the club per week
    #      - For the "Qualification" predictor, "Bachelor" and "Bachelor's" coding to be aligned as one value, as well as the 
    #        "Master" and "Master's" coding is to be aligned too

    Some findings on the EDA includes:
    
    The proportion of "Attrition" and "No Attrition" is unbalanced. The "Attrition" contributed to 17%, with the "No Attrition" 
    made up the remaining 87% of the sampling. The unbalanced might result to a 'weak' model in predicting Attrition.
    
    The models are supervised (with labelled categorical response), for classification of "Attrition" or"No Attrition",
    on numerical and categorical predictors.
    
    Example of suitable algorithms include KNN, Decision Tree , Logistics Regression, etc.
    
(2) MLP_Submit.py
    This is on Task 2.
    
    Below is the program header:
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
   
    
    A total of 5 models, on Knn and Decision Tree were fitted and validated.
    Below is '(10) Summary and interpretation':
    
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

(3) terminal_log.txt
    This is generated when the 'MLP_submitted.py' is executed.
    It contains the confusion matrix, classification report for all the 5 models trained and tested.
    
(4) X.Knn_model_1_Error_and_KNN_graph.png,  X.Knn_model_2_Error_and_KNN_graph.png,  X.Knn_model_3_Error_and_KNN_graph.png
    These are the graphs generated when 'MLP_submitted.py' is executed.
    These graphs plotted the mean error (y-axis), across a range of 1 to 20 KNN neigbours. 
    The plots are useful in determining the optimised KNN value.
    
(5) X_Tree_Plot_Model_4.png, X_Tree_Plot_Model_5.png
    These are the decision trees fitted for model 4 and 5.
    On the plots, it explain the predictors used, starting from the top of the tree, with the Gini rating at each node.
    The plots are useful in explaining the significance of the predictors.
    
    
    


    
