import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score,roc_auc_score,roc_curve,classification_report,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


#We read the data
covid = pd.read_csv('Covid Dataset.csv')

#Since the data is a categorical variable, LabelEncoding() is used to convert it to a numeric variable.
e=LabelEncoder()
covid['Breathing Problem']=e.fit_transform(covid['Breathing Problem'])
covid['Fever']=e.fit_transform(covid['Fever'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Running Nose']=e.fit_transform(covid['Running Nose'])
covid['Asthma']=e.fit_transform(covid['Asthma'])
covid['Chronic Lung Disease']=e.fit_transform(covid['Chronic Lung Disease'])
covid['Headache']=e.fit_transform(covid['Headache'])
covid['Heart Disease']=e.fit_transform(covid['Heart Disease'])
covid['Diabetes']=e.fit_transform(covid['Diabetes'])
covid['Hyper Tension']=e.fit_transform(covid['Hyper Tension'])
covid['Abroad travel']=e.fit_transform(covid['Abroad travel'])
covid['Contact with COVID Patient']=e.fit_transform(covid['Contact with COVID Patient'])
covid['Attended Large Gathering']=e.fit_transform(covid['Attended Large Gathering'])
covid['Visited Public Exposed Places']=e.fit_transform(covid['Visited Public Exposed Places'])
covid['Family working in Public Exposed Places']=e.fit_transform(covid['Family working in Public Exposed Places'])
covid['Wearing Masks']=e.fit_transform(covid['Wearing Masks'])
covid['Sanitization from Market']=e.fit_transform(covid['Sanitization from Market'])
covid['COVID-19']=e.fit_transform(covid['COVID-19'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Gastrointestinal ']=e.fit_transform(covid['Gastrointestinal '])
covid['Fatigue ']=e.fit_transform(covid['Fatigue '])


# As explained earlier, the effect of two variables on the dependent variable is zero during data preprocessing.
covid = covid.drop(["Sanitization from Market","Wearing Masks"],axis = 1)

# We separate the dependent and independent variables in the data set.
x=covid.drop('COVID-19',axis=1)
y=covid[['COVID-19']]
x.head()

# As explained before, the Chi-Squared method was chosen as Feature Selection Method and the 10 highest correlated independent variables were selected.
x_chi = x[["Abroad travel","Attended Large Gathering","Sore throat","Breathing Problem","Contact with COVID Patient",
               "Dry Cough","Fever","Family working in Public Exposed Places","Visited Public Exposed Places","Hyper Tension"]]

# The train and test data are set via the "hazirlik" function.
def hazirlik(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state = 0)
    modelem(X_train, X_test, y_train, y_test)

# In the study, the most optimal hyperparameters were found by using hyperparameter optimization methods for 5 different ML models.
# ML models were established with these hyperparameters and the KNN method with the highest success rate was selected. It was then deployed to Heroku, that is, Ai Server.
# How the ML model is deployed will not be explained here. And only an ML model will be shown as an example.

def modelem(X_train, X_test, y_train, y_test):
    yeni_bir = list()  
    #2 K-Nearest Neighbour Yontemi #n_neighbors = 8 değiştikce o da değişiyor
    knn_model = KNeighborsClassifier(n_neighbors = 8).fit(X_train,y_train)
    knn_model.get_params(deep = True)
    y_pred = knn_model.predict(X_test)
    yeni_bir.append("Bu K-NN : " + str(accuracy_score(y_pred,y_test)))
    

 
# To see our the accuracy rate of ML model
hazirlik(x_chi,y)
    