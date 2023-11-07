from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        training_and_testing_models(df)

def training_and_testing_models(df):

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    X = df[feature_names]
    y = df['y']

    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_model_pipeline.fit(X_train,y_train)
    rf_model_pipeline.fit(X_train,y_train)
    svm_model_pipeline.fit(X_train,y_train)
    xgb_model_pipeline.fit(X_train,y_train)

    print(f'######### Testing modelli #########')
    validate(lr_model_pipeline,'lr',X_test,y_test,True)
    validate(rf_model_pipeline,'rf',X_test,y_test)
    validate(svm_model_pipeline,'svm',X_test,y_test)
    validate(xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_bank_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_bank_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_bank_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_bank_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')


def validate(ml_model,ml_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"

    with  open(f"./reports/std_models/bank_metrics_report.txt",open_type) as f:
        f.write(f"{ml_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/std/bank_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')


start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)