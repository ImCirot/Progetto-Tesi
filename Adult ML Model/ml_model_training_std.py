import numpy as np 
import pandas as pd 
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
import pickle
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset e drop features superflue

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # drop ID dal dataset
    df.drop('ID',inplace=True,axis=1)

    training_model(df)


def training_model(dataset):
    ## funzione di apprendimento del modello sul dataset

    # setting nomi features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo il nome della feature target dalla lista nomi features
    features.remove('salary')

    # setting nome target feature
    target = ['salary']

    # setting dataset features
    X = dataset[features]

    # setting dataset target feature
    y = dataset[target]

    # costruiamo il modello standard tramite pipeline contenente uno scaler per la normalizzazione dati e un regressore
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # training del modello base sul training set 
    print(f'######### Training modelli #########')
    lr_model_pipeline.fit(X_train,y_train.values.ravel())
    rf_model_pipeline.fit(X_train,y_train.values.ravel())
    svm_model_pipeline.fit(X_train,y_train.values.ravel())
    xgb_model_pipeline.fit(X_train,y_train.values.ravel())

    # calcolo metriche di valutazione sul modello
    print(f'######### Testing modelli #########')
    validate(lr_model_pipeline,'lr',X_test,y_test,True)
    validate(rf_model_pipeline,'rf',X_test,y_test)
    validate(svm_model_pipeline,'svm',X_test,y_test)
    validate(xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_adult_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_adult_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_adult_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_adult_model.sav','wb'))
  
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')
    
def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/std_models/adult_metrics_report.txt",open_type) as f:
        f.write(f'{model_type}\n')
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/std/adult_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)