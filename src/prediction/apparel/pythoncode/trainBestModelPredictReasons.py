# import modules
import pandas as pd
import numpy as np
from sklearn.externals import joblib

#read metrics file
modelMetrics = pd.read_csv("src/prediction/apparel/csv/metricsData1.csv")
//////////////////////////////////


#Read data from the DB
dbConnection = DBConnection()
readDataSQL="SELECT `ID`,`Name`,`Career Growth`,`JoinedYear`,`Tenure`,`Age`,`Maritial Status`,`Total Salary`,`Promotions`,`Gender`,`Working Hours`,`Experience`,`Performance Rating`,`No.of Leaves`,`Participation of Activities` FROM `emppredict` "
#print("Apparel Predict dataset")
apperalDataPredict = dbConnection.readDataSet(readDataSQL)
#print(apperalData1);

c = csv.writer(open("src/prediction/apparel/csv/ApperalPredict.csv","w",newline=''))

c.writerow(["ID", "Name", "Career Growth", "JoinedYear", "Tenure", "Age", "Maritial Status", "Total Salary", "Promotions", "Training", "Gender" ,"Working Hours", "Experience", "Performance Rating", "No.of Leaves","Participation of Activities","churn"])

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
                x["Participation of Activities"],
                x["churn"]])

apperalDataPredict_csv = pd.read_csv("src/prediction/apparel/csv/ApperalPredict.csv")
#apperalData = pd.read_csv("src/prediction/apparel/csv/ApperalDataSet.csv")












////////////////////////


columns = apperalDataPredict_csv.columns.tolist()
# have to use only numeric values to the model
columns = [c for c in columns if
           c not in ["ID", "Name"]]

#print(modelMetrics.columns)
#print(modelMetrics)
#accuracy=modelMetrics["Accuracy"]
#print(modelMetrics["Accuracy"])
i=modelMetrics["Accuracy"].argmax()
model=modelMetrics["Unnamed: 0"][i]

modelLogisticRegressionFile='LogisticRegression.joblib.pkl'
modelDesicionTreeFile='DesicionTree.joblib.pkl'
modelSVMFile='SVM.joblib.pkl'
modelRandomForestClassifierFile='RandomForestClassifier.joblib.pkl'
modelKNeighborsClassifierFile='KNeighbors.joblib.pkl'

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






