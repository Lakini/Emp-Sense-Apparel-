# import modules
import pandas as pd
import numpy as np
import csv
from DBAccess import DBConnection
from sklearn.externals import joblib

#read metrics file
modelMetrics = pd.read_csv("src/prediction/apparel/csv/modelManual.csv")
#Read data from the DB
dbConnection = DBConnection()
readDataSQL="SELECT `ID`,`Name`,`Career Growth`,`JoinedYear`,`Tenure`,`Age`,`Maritial Status`,`Total Salary`,`Promotions`,`Training`,`Gender`,`Working Hours`,`Experience`,`Performance Rating`,`No.of Leaves`,`Participation of Activities` FROM `emppredict` "
print("Apparel Predict dataset")
apperalDataPredict = dbConnection.readDataSet(readDataSQL)


c = csv.writer(open("src/prediction/apparel/csv/ApperalPredict.csv","w",newline=''))

c.writerow(["ID", "Name", "Career Growth", "JoinedYear", "Tenure", "Age", "Maritial Status", "Total Salary", "Promotions","Training", "Gender" ,"Working Hours", "Experience", "Performance Rating", "No.of Leaves","Participation of Activities"])

for x in apperalDataPredict:
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
                x["Participation of Activities"],])

apperalDataToPredict= pd.read_csv("src/prediction/apparel/csv/ApperalPredict.csv")
#apperalData = pd.read_csv("src/prediction/apparel/csv/ApperalDataSet.csv")

columns = apperalDataToPredict.columns.tolist()

# have to use only numeric values to the model
columns = [c for c in columns if
           c not in ["ID", "Name"]]

model=modelMetrics["modelName"][0]
print("model name")
print(model)

modelLogisticRegressionFile='LogisticRegression.joblib.pkl'
modelDesicionTreeFile='DesicionTree.joblib.pkl'
modelSVMFile='SVM.joblib.pkl'
modelRandomForestClassifierFile='RandomForestClassifier.joblib.pkl'
modelKNeighborsClassifierFile='KNeighbors.joblib.pkl'

print(apperalDataToPredict[columns])

#Accessing corresponding model

if(model=="LogisticReg"):
    logisticRegressionModel = joblib.load(modelLogisticRegressionFile)
    predictions=logisticRegressionModel.predict(apperalDataToPredict[columns])
elif(model=="SVMC"):
    svmModel = joblib.load(modelSVMFile)
    predictions=svmModel.predict(apperalDataToPredict[columns])
elif (model == "DecisionTree"):
    desicionTreeModel = joblib.load(modelDesicionTreeFile)
    predictions=desicionTreeModel.predict(apperalDataToPredict[columns])
elif (model == "RandomForest"):
    randomForestModel = joblib.load(modelRandomForestClassifierFile)
    predictions=randomForestModel.predict(apperalDataToPredict[columns])
elif (model == "kNN9"):
    knnModel = joblib.load(modelKNeighborsClassifierFile)
    predictions=knnModel.predict(apperalDataToPredict[columns])

    #df1['e'] = Series(np.random.randn(sLength), index=df1.index)
#predictions['EmployeeName']= pd.Series(apperalDataToPredict["Name"], index=predictions.index)
#print(predictions)

IDs = np.array(apperalDataToPredict.ID)
names= np.array(apperalDataToPredict.Name)
print(IDs)
print(predictions)
DAT =  np.column_stack((IDs, names, predictions))
np.savetxt('src/prediction/apparel/csv/predictedchurn.csv',DAT, delimiter=" ", fmt="%s")
