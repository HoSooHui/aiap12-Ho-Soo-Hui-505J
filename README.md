# aiap12-Ho-Soo-Hui-505J
AIAP Assessment

Full Name: Ho Soo Hui
Email address (stated in the Application Form) : Soohui-ho@hotmail.com

In the 'Main' branch, there are the below commited files:

(i) eda.ipynb. 
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
    
    For each of the 7 processes, a summary/observations could be found at the end of the process, and before the begining of the next process.
    
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
    
    Example of suitable algorithms includes KNN, Decision Tree , Logistics Regression, etc.
    
(2) MLP_Submit.py
    This is on Task 2. - Exploratory Data Analysis.
    


