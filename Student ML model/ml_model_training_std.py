from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Student Dataset/dataset.csv')

    # drop ID dal dataframe
    df.drop('ID', inplace=True,axis=1)

    # richiamo funzione di training e testing dei modelli
    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        training_testing_models(df)

def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli
    feature_names = dataset.columns.tolist()

    feature_names.remove('Target')

    X = dataset[feature_names]
    y = dataset['Target']


    # settiamo i nostri modelli sul dataset originale
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_model_pipeline.fit(X_train,y_train.values.ravel())
    rf_model_pipeline.fit(X_train,y_train.values.ravel())
    svm_model_pipeline.fit(X_train,y_train.values.ravel())
    xgb_model_pipeline.fit(X_train,y_train.values.ravel())

    print(f'######### Testing modelli #########')
    validate(lr_model_pipeline,'lr', X_test, y_test,True)
    validate(rf_model_pipeline,'rf',X_test,y_test)
    validate(svm_model_pipeline,'svm',X_test,y_test)
    validate(xgb_model_pipeline,'xgb',X_test,y_test)
    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_student_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_student_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_student_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_student_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/std_models/student_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {accuracy}")
        f.write(f'\nROC-AUC score: {auc_score}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/std/student_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')


start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)