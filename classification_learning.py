import pandas as pd
import csv as csv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


# LOAD DATA INTO DATAFRAME
train_df = pd.read_csv("input/train.csv", header=0)
test_df  = pd.read_csv("input/test.csv", header=0)

# USEFUL INFORMATION
print train_df.info()
print test_df.info()

# print number of missing values
print train_df.apply(lambda x: sum(x.isnull()), axis=0)
print test_df.apply(lambda x: sum(x.isnull()), axis=0) 

train_header = list(train_df.columns.values)
test_header = list(test_df.columns.values)

# explore distributions

sns.boxplot(x=train_df['Education'], y = train_df['ApplicantIncome'], hue = train_df['Gender'])
sns.plt.show()

train_df['CoapplicantIncome'].hist(bins=50)
plt.xlabel('Coapplicant income')
plt.show()

train_df['LoanAmount'].hist(bins=50)
plt.xlabel('loan amount')
plt.show()

train_df['Loan_Amount_Term'].hist(bins=50)
plt.xlabel('loan amount term')
plt.show()

print "there are extreme values in Applicant income and loan amount"


# DATA CLEANUP

def map_data(df, column, mapping):
    """
    map column in dataframe
    """
    df[column] = df[column].map(mapping).dropna().astype(int)
    

def fill_most_common(df, column):
    """
    fill NaN values in column in df by the most common entry
    """
    
    if len(df[column][df[column].isnull()]) > 0:    
        df[column][df[column].isnull()] = df[column].dropna().mode().values    


def fill_mean(df, column):    
    """
    fill NaN values in column in df by the mean of the column
    """   

    mean_value = df[column].dropna().mean()

    if len(df[column][df[column].isnull()]) > 0:
        df.loc[(df[column].isnull()), column] = mean_value
        

# gender
map_gender = {'Male':1, 'Female':0}    
map_data(train_df, 'Gender', map_gender)
map_data(test_df, 'Gender', map_gender)

fill_most_common(train_df, 'Gender')
fill_most_common(test_df, 'Gender')

# married
map_married = {'Yes':1, 'No':0}
map_data(train_df, 'Married', map_married)
map_data(test_df, 'Married', map_married)

fill_most_common(train_df, 'Married')

# dependents
print train_df['Dependents'].unique()

map_dependents = {'0':0, '1':1, '2':2, '3+':3}
map_data(train_df, 'Dependents', map_dependents)
map_data(test_df, 'Dependents', map_dependents)

fill_most_common(train_df, 'Dependents')
fill_most_common(test_df, 'Dependents')

# education
map_education = {'Not Graduate':0, 'Graduate':1}
map_data(train_df, 'Education', map_education)
map_data(test_df, 'Education', map_education)

# self-employed
map_self = {'No':0, 'Yes':1}
map_data(train_df, 'Self_Employed', map_self)
map_data(test_df, 'Self_Employed', map_self)

fill_most_common(train_df, 'Self_Employed')
fill_most_common(test_df, 'Self_Employed')

# loan amount
fill_mean(train_df, 'LoanAmount')
fill_mean(test_df, 'LoanAmount')

# log transformation to cancel extreme values
train_df['LoanAmount_log'] = np.log(train_df['LoanAmount'])
test_df['LoanAmount_log'] = np.log(test_df['LoanAmount'])

train_df['LoanAmount_log'].hist(bins=20)
test_df['LoanAmount_log'].hist(bins=20)
plt.show()

# applicant and co applicant income
# combine them in total income and apply log to cancel extreme values

def calc_total_income(df):

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])

calc_total_income(train_df)
calc_total_income(test_df)

# loan amount term

fill_mean(train_df, 'Loan_Amount_Term')
fill_mean(test_df, 'Loan_Amount_Term')

# credit history
print train_df['Credit_History'].unique()

fill_most_common(train_df, 'Credit_History')
fill_most_common(test_df, 'Credit_History')

# property area
print train_df['Property_Area'].unique()

map_property = {'Rural':0, 'Semiurban':1, 'Urban':2}
map_data(train_df, 'Property_Area', map_property)
map_data(test_df, 'Property_Area', map_property)

# loan status
map_status = {'N':0, 'Y':1}
map_data(train_df, 'Loan_Status', map_status)

# drop un-insightful columns
train_df = train_df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)

idx = test_df['Loan_ID'].values
test_df  = test_df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)

# reorder column: put LoanStatus at the beginning

col = train_df['Loan_Status']
train_df.drop(labels=['Loan_Status'], axis=1, inplace = True)
train_df.insert(0, 'Loan_Status', col)

train_header = list(train_df.columns.values)
test_header = list(test_df.columns.values)


# MACHINE LEARNING

def classification_model(model, train_data, test_data, predictors, outcome):
    """
    make a classification model and accessing performance
    model: eg. model = LogisticRegression()
    train data: training dataframe
    test_data: test dataframe    
    predictor: list of column labels used to train the model
    outcome: column label for the objective to reach
    """
    #Fit the model:
    model.fit(train_data[predictors], train_data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(train_data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions, train_data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # print score
    score = model.score(train_data[predictors], train_data[outcome])
    print "score: %s" % "{0:.3%}".format(score)

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(train_data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (train_data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = train_data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(train_data[predictors].iloc[test,:], train_data[outcome].iloc[test]))
 
    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    #Fit the model again so that it can be refered outside the function:
    model.fit(train_data[predictors], train_data[outcome]) 
    
    # predict on test set
    out = model.predict(test_data[predictors])
    
    return out


outcome_var = 'Loan_Status'

# logistic regression
model = LogisticRegression()
predictor_var = ['Credit_History']
out_logreg = classification_model(model, train_df, test_df, predictor_var, outcome_var)

predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
out_logreg2 = classification_model(model, train_df, test_df, predictor_var, outcome_var)

# decision tree
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
out_dectre = classification_model(model, train_df, test_df, predictor_var, outcome_var)

predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
out_dectre2 = classification_model(model, train_df, test_df, predictor_var, outcome_var)

# random forest classifier
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
out_rfc = classification_model(model, train_df, test_df, predictor_var, outcome_var)

# create a series with feature importances !!!
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print featimp

# lets use only the first 3 best variables to limit overfitting
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
out_rfc2 = classification_model(model, train_df, test_df, predictor_var, outcome_var)

# SVM
model = SVC()
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
out_svc = classification_model(model, train_df, test_df, predictor_var, outcome_var)

## knn classifier
#model = KNeighborsClassifier(n_neighbors = 5)
#predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
#out_knn = classification_model(model, train_df, test_df, predictor_var, outcome_var)

# credit history model: if no credit history, no loan]

out_credit = test_df['Credit_History']

# format output        

def format_output(out):
    """
    format output value according to the rule: 0 = 'N' and 1 = 'Y'
    """
    
    formatted_output = []
    
    for i in out:
        if i == 1:
            formatted_output.append('Y')
        else:
            formatted_output.append('N')            

    return formatted_output


out_logreg = format_output(out_logreg)
out_logreg2 = format_output(out_logreg2)

out_dectre = format_output(out_dectre)
out_dectre2 = format_output(out_dectre2)

out_rfc = format_output(out_rfc)
out_rfc2 = format_output(out_rfc2)

out_svc = format_output(out_svc)
#out_knn = format_output(out_knn)
out_credit = format_output(out_credit)

# write out predictions 

def write_output(filename, idx, out):
    """
    write result of machine learning to filename: 1st column is idx and 
    2nd column is out
    """        
    predictions_file = open(filename, 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['Loan_ID', 'Loan_Status'])
    open_file_object.writerows(zip(idx, out))
    predictions_file.close()         


write_output("output/logistic_regression.csv", idx, out_logreg)
write_output("output/logistic_regression2.csv", idx, out_logreg2)

write_output("output/random_forest.csv", idx, out_rfc)
write_output("output/random_forest2.csv", idx, out_rfc2)

write_output("output/decision_tree.csv", idx, out_dectre)
write_output("output/decision_tree2.csv", idx, out_dectre2)

write_output("output/svm.csv", idx, out_svc)
#write_output(output/knn.csv", idx, out_knn)

write_output("output/credit_model.csv", idx, out_credit)
