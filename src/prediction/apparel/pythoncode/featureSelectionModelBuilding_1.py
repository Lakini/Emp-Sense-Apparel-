#import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import threading
from sklearn import preprocessing
from sklearn.cross_validation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from DBAccess import DBConnection
#C:\Users\Lakini\Documents\CDAP-mypart\ApparelPrediction\src\prediction\apparel\pythoncode\featureImportance.py
#from com.dbconn.DBAccess import DBConnection

#read data file
#apperalData = pd.read_csv("src/prediction/apparel/csv/ApperalDataSet.csv")

#Read data from the DB
dbConnection = DBConnection()
readDataSQL="SELECT `ID`,`Name`,`Career Growth`,`JoinedYear`,`Tenure`,`Age`,`Maritial Status`,`Total Salary`,`Promotions`,`Training`,`Gender`,`Working Hours`,`Experience`,`Performance Rating`,`No.of Leaves`,`Participation of Activities`,`churn` FROM `empapparel` "

#print("Apparel dataset")
apperalData1 = dbConnection.readDataSet(readDataSQL)
#print(apperalData1);

c = csv.writer(open("src/prediction/apparel/csv/Lakini.csv","w",newline=''))

c.writerow(["ID", "Name", "Career Growth", "JoinedYear", "Tenure", "Age", "Maritial Status", "Total Salary", "Promotions", "Training", "Gender" ,"Working Hours", "Experience", "Performance Rating", "No.of Leaves","Participation of Activities","churn"])

for x in apperalData1:
    c.writerow([x["ID"], 
                x["Name"], 
                x["Career Growth"], 
                x["JoinedYear"],
                x["Tenure"],
                x["Age"],
                x["Maritial Status"],
                x["Total Salary"],
                x["Promotions"],
                x["Training"],
                x["Gender"],
                x["Working Hours"],
                x["Experience"],
                x["Performance Rating"],
                x["No.of Leaves"],
                x["Participation of Activities"],
                x["churn"]])

apperalData_csv = pd.read_csv("src/prediction/apparel/csv/Lakini.csv")
#apperalData = pd.read_csv("src/prediction/apparel/csv/ApperalDataSet.csv")
    

#print("----------------------csv from DB readigs-----------------------")
#print(apperalData_csv)



#X_train, X_test, y_train, y_test = train_test_split(apperalData_csv[columns], apperalData_csv[target], test_size = 0.4, random_state = 42)

#apperalDataToPredict = pd.read_csv("src/prediction/apparel/csv/ApperalDataSetToPredict.csv")

# get the columns in the csv
global columns
####Basic Step###################################################################

columns = apperalData_csv.columns.tolist()
# Remove Unwanted Labels to predict labels.But here we have to vectorised the data.
# have to use only numeric values to the model
columns = [c for c in columns if
           c not in ["ID", "Name", "churn"]]
# Set the predicted target to Churn
target = "churn"
# Generate the training set.  Set random_state to be able to replicate results.
#distribute data for x and y matrices.
X_train, X_test, y_train, y_test = train_test_split(apperalData_csv[columns], apperalData_csv[target], test_size = 0.4, random_state = 42)

# create a new dataframe with scaled features (if feature is not binary)
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#######################################end of basic step####################################################
# define a method for retrieving roc parameters and train the model
#########################Trained the model anf get the output######################################

#####With unscaled parameters
#########train modelLogisticRegression
def trainLogisticRegression():
    modelLogisticRegression=LogisticRegression()
    #set the number of features to 10
    rfelogisticReg=RFE(modelLogisticRegression,10)
    rfelogisticReg=rfelogisticReg.fit(X_train, y_train)
    print("Feature Importance of Logistic Regression Model")
    print(rfelogisticReg.support_)
    print(rfelogisticReg.ranking_)
    modelLogisticRegression.fit(X_train, y_train)
    return modelLogisticRegression

####train modelDesion tree
def trainDesicionTreeClassifier():
    modelDesicionTree=DecisionTreeClassifier(max_depth=5)
    # set the number of features to 10
    rfedecisiontree = RFE(modelDesicionTree, 10)
    rfedecisiontree = rfedecisiontree.fit(X_train, y_train)
    print("Feature Importance of Decision Tree Model")
    print(rfedecisiontree.support_)
    print(rfedecisiontree.ranking_)
    modelDesicionTree.fit(X_train, y_train)
    return modelDesicionTree

####train modelSVM
def trainSVM():
    modelSVM=SVC(probability=True)
    # # set the number of features to 10
    # rfesvm= RFE(modelSVM, 10)
    # rfesvm = rfesvm.fit(X_train, y_train)
    # print("Feature Importance of Decision Tree Model")
    # print(rfesvm.support_)
    # print(rfesvm.ranking_)
    modelSVM.fit(X_train, y_train)
    return modelSVM

####train modelRandomForestClassifier
def trainRandomForestClassifier():
    modelRandomForestClassifier=RandomForestClassifier()
    # set the number of features to 10
    # rferandomForest = RFE(modelRandomForestClassifier, 10)
    # rferandomForest = rferandomForest.fit(X_train, y_train)
    # print("Feature Importance of Decision Tree Model")
    # print(rferandomForest.support_)
    # print(rferandomForest.ranking_)
    modelRandomForestClassifier.fit(X_train, y_train)
    return modelRandomForestClassifier

####train modelKNeighborsClassifier
def trainKNeighborsClassifier():
    modelKNeighborsClassifier=KNeighborsClassifier(n_neighbors=9)
    # # set the number of features to 10
    # rfeKneighbirs = RFE(modelKNeighborsClassifier, 10)
    # rferandomForest = rfeKneighbirs.fit(X_train, y_train)
    # print("Feature Importance of Decision Tree Model")
    # print(rferandomForest.support_)
    # print(rferandomForest.ranking_)
    modelKNeighborsClassifier.fit(X_train, y_train)
    return modelKNeighborsClassifier


##With scaled Parameters
########train modelLogisticRegression
def trainLogisticRegressionScaled():
    modelLogisticRegressionScaled=LogisticRegression()
    modelLogisticRegressionScaled.fit(X_train_std, y_train)
    return modelLogisticRegressionScaled

####train modelDesion tree
def trainDesicionTreeClassifierScaled():
    modelDesicionTreeScaled=DecisionTreeClassifier(max_depth=5)
    modelDesicionTreeScaled.fit(X_train_std, y_train)
    return modelDesicionTreeScaled

####train modelSVM
def trainSVMScaled():
    modelSVMScaled=SVC(probability=True)
    modelSVMScaled.fit(X_train_std,y_train)
    return modelSVMScaled

####train modelRandomForestClassifier
def trainRandomForestClassifierScaled():
    modelRandomForestClassifierScaled=RandomForestClassifier()
    modelRandomForestClassifierScaled.fit(X_train_std,y_train)
    return modelRandomForestClassifierScaled

####train modelKNeighborsClassifier
def trainKNeighborsClassifierScaled():
    modelKNeighborsClassifierScaled=KNeighborsClassifier(n_neighbors=9)
    modelKNeighborsClassifierScaled.fit(X_train_std, y_train)
    return modelKNeighborsClassifierScaled

#making models accessible
modelLogisticRegression=trainLogisticRegression()
modelDesicionTree=trainDesicionTreeClassifier()
modelSVM=trainSVM()
modelRandomForestClassifier=trainRandomForestClassifier()
modelKNeighborsClassifier=trainKNeighborsClassifier()
modelLogisticRegressionScaled=trainLogisticRegressionScaled()
modelDesicionTreeScaled=trainDesicionTreeClassifierScaled()
modelSVMScaled=trainSVMScaled()
modelRandomForestClassifierScaled=trainRandomForestClassifierScaled()
modelKNeighborsClassifierScaled=trainKNeighborsClassifierScaled()

#save created models in a joblib
modelLogisticRegressionFile='LogisticRegression.joblib.pkl'
_ = joblib.dump(modelLogisticRegression, modelLogisticRegressionFile, compress=9)

modelDesicionTreeFile='DesicionTree.joblib.pkl'
_ = joblib.dump(modelDesicionTree, modelDesicionTreeFile, compress=9)

modelSVMFile='SVM.joblib.pkl'
_ = joblib.dump(modelSVM, modelSVMFile, compress=9)

modelRandomForestClassifierFile='RandomForestClassifier.joblib.pkl'
_ = joblib.dump(modelRandomForestClassifier, modelRandomForestClassifierFile, compress=9)

modelKNeighborsClassifierFile='KNeighbors.joblib.pkl'
_ = joblib.dump(modelKNeighborsClassifier, modelKNeighborsClassifierFile, compress=9)

modelLogisticRegressionScaledFile='LogisticRegressionScaled.joblib.pkl'
_ = joblib.dump(modelLogisticRegressionScaled, modelLogisticRegressionScaledFile, compress=9)

modelDesicionTreeScaledFile='DesicionTreeScaled.joblib.pkl'
_ = joblib.dump(modelDesicionTreeScaled, modelDesicionTreeScaledFile, compress=9)

modelSVMScaledFile='SVMScaled.joblib.pkl'
_ = joblib.dump(modelSVMScaled, modelSVMScaledFile, compress=9)

modelRandomForestClassifierScaledFile='RandomForestClassifierScaled.joblib.pkl'
_ = joblib.dump(modelRandomForestClassifierScaled, modelRandomForestClassifierScaledFile, compress=9)

modelKNeighborsClassifierScaledFile='KNeighborsClassifierScaled.joblib.pkl'
_ = joblib.dump(modelKNeighborsClassifierScaled, modelKNeighborsClassifierScaledFile, compress=9)

#clf2 = joblib.load(filename)

# set modeling and plotting parameters to unscaled parameters
def roc_params_unscaled(name):
    if(name=="Logistic Regression"):
        test_pred = modelLogisticRegression.predict_proba(X_test)

    elif(name=="Decision Tree Classifier"):
        test_pred = modelDesicionTree.predict_proba(X_test)

    elif(name=="Support Vector Machine"):
        test_pred = modelSVM.predict_proba(X_test)

    elif(name=="Random Forest Classifier"):
        test_pred = modelRandomForestClassifier.predict_proba(X_test)

    elif(name=="K Neighbors (9)"):
        test_pred = modelKNeighborsClassifier.predict_proba(X_test)

    # return the roc parameters
    fpr, tpr, _ = roc_curve(y_test, test_pred[:, 1])
    roc_params = [fpr, tpr]
    return roc_params

# # set modeling and plotting parameters to scaled parameters
def roc_params_scaled(name):
    if(name=="Logistic Regression"):
        test_pred = modelLogisticRegressionScaled.predict_proba(X_test_std)

    elif(name=="Decision Tree Classifier"):
        test_pred = modelDesicionTreeScaled.predict_proba(X_test_std)

    elif(name=="Support Vector Machine"):
        test_pred = modelSVMScaled.predict_proba(X_test_std)

    elif(name=="Random Forest Classifier"):
        test_pred = modelRandomForestClassifierScaled.predict_proba(X_test_std)

    elif(name=="K Neighbors (9)"):
        test_pred = modelKNeighborsClassifierScaled.predict_proba(X_test_std)

    # return the roc parameters
    fpr, tpr, _ = roc_curve(y_test, test_pred[:, 1])
    roc_params = [fpr, tpr]
    return roc_params

# #Call functions in the modelBuildeing class.
names = ["Logistic Regression", "Decision Tree Classifier","Support Vector Machine",
        "Random Forest Classifier", "K Neighbors (9)"]

colors = ["b", "k", "g", "r","y"]

# plot roc curves
plt.figure(figsize = (18,12))

#########Show seperate ROC graphs for all.
for i, model in enumerate(names):
    j=i+1
    plt.subplot(2,3, j)
    plt.title(names[i], fontsize = 10)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    # plot parameters using unscaled predictors
    params =roc_params_unscaled(names[i])
    plt.plot(params[0], params[1], colors[i] + "--", linewidth = 2, label = "Unscaled")

    # plot parameters using scaled predictors
    scaled_params = roc_params_scaled(names[i])
    plt.plot(scaled_params[0], scaled_params[1], colors[i], linewidth = 2, label = "Scaled")
    plt.legend(loc = 4)

plt.tight_layout()
plt.savefig('allGraphs.png')
#plt.show()

########ROC curves in one diagram####################################
# plot ROC curves of estimators on scaled predictors
plt.figure(figsize = (8, 8))
plt.title("ROC curves for models using scaled predictors", fontsize = 20)

for i,j in enumerate(names):
    scaled_params = roc_params_scaled(names[i])
    plt.plot(scaled_params[0], scaled_params[1], colors[i], label = names[i])

plt.legend(loc = 4, prop={'size': 18})
plt.tight_layout()
plt.savefig("ROCChart.png")

##############################get the Accuracy,Precision,Recall and F1 Scores############
scores = [accuracy_score, precision_score, recall_score, f1_score]

def get_metrics(X_test):
    # create empty lists
    LogisticReg = []
    SVMC = []
    DecisionTree = []
    RandomForest = []
    kNN9 = []

    # list of lists
    lists = [LogisticReg, SVMC, DecisionTree, RandomForest, kNN9]

    # populate lists with scores of each scoring method
    for i, model in enumerate(names):
        for score in scores:
            name = model

            if (name == "Logistic Regression"):
                pred = modelLogisticRegression.predict(X_test)

            elif (name == "Decision Tree Classifier"):
                pred = modelDesicionTree.predict(X_test)

            elif (name == "Support Vector Machine"):
                pred = modelSVM.predict(X_test)

            elif (name == "Random Forest Classifier"):
                pred = modelRandomForestClassifier.predict(X_test)

            elif (name == "K Neighbors (9)"):
                pred = modelKNeighborsClassifier.predict(X_test)

            lists[i].append(score(y_test, pred))

    # create a dataframe which aggregates the lists
    scores_df = pd.DataFrame(data = [LogisticReg, SVMC, DecisionTree, RandomForest, kNN9])
    scores_df.index = ["LogisticReg", "SVMC", "DecisionTree", "RandomForest", "kNN9"]
    scores_df.columns = ["Accuracy", "Precision", "Recall", "F1"]
    return scores_df

metricsData=get_metrics(X_test)
# model_names=["LogisticReg","SVMC","DecisionTree","RandomForest","kNN9"]

#x=DBConnection()
# for i in range(5):
#     sql= "INSERT INTO `modeldetails`(`Model`, `Accuracy`, `Precision_details`, `Recall`, `F1`) VALUES (%s, %d, %d, %d, %d)" %(model_names[i],metricsData["Accuracy"][i],metricsData["Precision"][i],metricsData["Recall"][i],metricsData["F1"][i])
#     x.insert_update(sql)

metricsData.to_csv("src/prediction/apparel/csv/metricsData1.csv");

modelMetrics = pd.read_csv("src/prediction/apparel/csv/metricsData1.csv")
i=modelMetrics["Accuracy"].argmax()
model=modelMetrics["Unnamed: 0"][i]
modelAccuracy=modelMetrics["Accuracy"][i]

x=DBConnection()
#x=DBConnection()
sql1="UPDATE `modeldetails` SET `Model`='%s',`Accuracy`=%d WHERE 1" %(model,modelAccuracy)
#sql= "INSERT INTO `modeldetails`(`Model`, `Accuracy`) VALUES (%s, %d)" %(model_names[i],metricsData["Accuracy"][i],metricsData["Precision"][i],metricsData["Recall"][i],metricsData["F1"][i])
x.insert_update(sql1)

