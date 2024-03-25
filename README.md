# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1 
Import the required packages and print the present data.
### Step-2
Print the placement data and salary data.
### Step-3
Find the null and duplicate values.
### Step-4
Using logistic regression find the predicted values of accuracy , confusion matrices.
### Step-5
Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHYAM.S
RegisterNumber: 212223240156 
*/
import pandas as pd

data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print(data1)

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0 )

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix( y_test,y_pred)

from sklearn.metrics import classification_report
classification_report1 =classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Placement Data:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/3e81827b-fa01-4637-b9fc-57b8d011c3d5)
## Print Data:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/01d024f7-5670-44d1-9a88-3f2e3d1d448d)

![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/652231b3-9eb1-4073-9104-e17917cf6c4d)
## Y_prediction array:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/17166c5e-5a33-47cf-948a-f0d3583f23a3)
## Accuracy value:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/61183f89-ae93-4ee8-859c-3ac41ec49944)
## Confusion array:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/818669d2-630e-498d-9921-d4f6517d3172)
## Classification Report:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/6332e2a3-9590-42aa-b41e-c2c4e523b31b)
## Prediction of LR:
![image](https://github.com/SridharShyam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144871368/9c0f8556-358c-45d7-9c3d-6b3de69220cc)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
