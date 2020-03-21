# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df=pd.read_csv(path)
# Code starts here
X=df.drop(['customerID','Churn'],1)
y=df['Churn'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state = 0)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges']=X_train['TotalCharges'].replace(' ',np.NaN).astype(float)
X_test['TotalCharges']=X_test['TotalCharges'].replace(' ',np.NaN).astype(float)
X_train['TotalCharges']=X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges']=X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())
X_train.isnull().sum()


cat_cols = X_train.select_dtypes(include='O').columns.tolist()

#Label encoding train data
for x in cat_cols:
    le = LabelEncoder()
    X_train[x] = le.fit_transform(X_train[x])
cate_cols = X_test.select_dtypes(include='O').columns.tolist()

#Label encoding train data
for x in cate_cols:
    le = LabelEncoder()
    X_test[x] = le.fit_transform(X_test[x])    
y_train=y_train.replace({'No':0, 'Yes':1})
y_test=y_test.replace({'No':0, 'Yes':1})


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train)
print(X_test)
print(y_train)
print(y_test)
ada_model=AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred=ada_model.predict(X_test)
ada_score=accuracy_score(y_test,y_pred)
ada_cm=confusion_matrix(y_test,y_pred)
ada_cr=classification_report(y_test,y_pred)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

#Initializing the model
xgb_model = XGBClassifier(random_state=0)

#Fitting the model on train data
xgb_model.fit(X_train,y_train)

#Making prediction on test data
y_pred = xgb_model.predict(X_test)

#Finding the accuracy score
xgb_score = accuracy_score(y_test,y_pred)
print("Accuracy: ",xgb_score)

#Finding the confusion matrix
xgb_cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n', xgb_cm)

#Finding the classification report
xgb_cr=classification_report(y_test,y_pred)
print('Classification report: \n', xgb_cr)


### GridSearch CV

#Initialsing Grid Search
clf = GridSearchCV(xgb_model, parameters)

#Fitting the model on train data
clf.fit(X_train,y_train)

#Making prediction on test data
y_pred = clf.predict(X_test)

#Finding the accuracy score
clf_score = accuracy_score(y_test,y_pred)
print("Accuracy: ",clf_score)

#Finding the confusion matrix
clf_cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n', clf_cm)

#Finding the classification report
clf_cr=classification_report(y_test,y_pred)
print('Classification report: \n', clf_cr)

#Code ends here


