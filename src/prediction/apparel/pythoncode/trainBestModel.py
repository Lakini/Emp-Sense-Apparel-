# import modules
import pandas as pd
import numpy as np
from sklearn.externals import joblib

#read metrics file
modelMetrics = pd.read_csv("src/prediction/apparel/csv/metricsData1.csv")
#read data to predict using the trained model
apperalDataToPredict = pd.read_csv("src/prediction/apparel/csv/ApperalDataSetToPredict.csv")

columns = apperalDataToPredict.columns.tolist()
# have to use only numeric values to the model
columns = [c for c in columns if
           c not in ["ID", "Name", "Basic Salary", "churn", "Health Status", "Recidency", "Past Job Role",
                     "Education", "Job Role"]]

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






