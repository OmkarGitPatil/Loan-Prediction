from flask import Flask,request,render_template,url_for
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\rfc_model.pkl','rb') as file:
    rfc_model=pkl.load(file)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\optimal_rfc_model.pkl','rb') as file1:
    optimal_rfc_model=pkl.load(file1)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\ai_scaler.pkl','rb') as file2:
    ai_scaler=pkl.load(file2)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\cai_scaler.pkl','rb') as file3:
    cai_scaler=pkl.load(file3)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\la_scaler.pkl','rb') as file4:
    la_scaler=pkl.load(file4)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Loan Prediction\artifacts\xgb_model.pkl','rb') as file5:
    xgb_model=pkl.load(file5)

app=Flask('__main__')

@app.route('/',)
def connect():
    return render_template('index.html')

@app.route('/input', methods=['GET','POST'])
def input():
    
    input_dict=request.form
    keys=list(input_dict.keys())
    values=list(input_dict.values())
    keys_values=list(zip(keys,values))

  #   Dependents, Education, Self_Employed, ApplicantIncome ,CoapplicantIncome,
  # LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Gender, Married
  # 
    # ['0', 'M', 'Y', '2', '2', '20000', '20000', '20000', '20000', '1', '0']
    array=np.zeros(13)

    array[0]=int(values[0])

    array[1]=int(values[3])
    array[2]=int(values[4])
    array[3]=ai_scaler.transform([[int(values[5])]])
    array[4]=cai_scaler.transform([[int(values[6])]])
    array[5]=la_scaler.transform([[int(values[7])]])
    array[6]=int(values[8])
    array[7]=int(values[9])
    array[8]=int(values[10])

    if values[1] == 'M':
        array[9]=0
        array[10]=1
    elif values[1] == 'F':
        array[9]=1
        array[10]=0

    if values[2] == 'Y':
        array[11]=0
        array[12]=1
    elif values[2] == 'N':
        array[11]=1
        array[12]=0

    # print(array)

    prediction=rfc_model.predict([array])
    if prediction == 1:
        output='Approved'
    elif prediction == 0:
        output='Denied'

    return render_template('display.html',Result=output,your_inputs=keys_values)


if __name__ == '__main__':
    app.run()