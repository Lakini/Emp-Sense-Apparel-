from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import *
from sklearn.cross_validation import *

#read data file
apperalData = pd.read_csv("src/prediction/apparel/csv/ApperalDataSet.csv")
apperalDataToPredict = pd.read_csv("src/prediction/apparel/csv/ApperalDataSetToPredict.csv")

# get the columns in the csv
global columns
####Basic Step###################################################################

columns = apperalData.columns.tolist()
# Remove Unwanted Labels to predict labels.But here we have to vectorised the data.
# have to use only numeric values to the model
columns = [c for c in columns if
           c not in ["ID", "Name", "Basic Salary", "churn", "Health Status", "Recidency", "Past Job Role",
                     "Education", "Job Role"]]
# Set the predicted target to Churn
target = "churn"
# Generate the training set.  Set random_state to be able to replicate results.
#distribute data for x and y matrices.
X_train, X_test, y_train, y_test = train_test_split(apperalData[columns], apperalData[target], test_size = 0.4, random_state = 42)

# create a new dataframe with scaled features (if feature is not binary)
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

##have to modify more##for all the algorithms