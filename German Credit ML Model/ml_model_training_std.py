import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset,StandardDataset
from aif360.metrics import ClassificationMetric
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep


def load_dataset():
    ## funzione di load del dataset
    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    df['Target'] = df['Target'].replace(2,0)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_and_testing_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(1)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(offline=True, country_iso_code="ITA")
def training_and_testing_model(df):
    ## Funzione per il training e testing del modello scelto
    features = df.columns.tolist()
    features.remove('Target')

    target = ['Target'] 

    sex_features = [
        'sex_A91','sex_A92','sex_A93','sex_A94'
    ]

    X = df[features]
    y = df[target]
    g = df[sex_features]

    # Creiamo due pipeline che effettuano delle ulteriori operazioni di scaling dei dati per addestriare il modello
    # in particolare la pipeline standard sarà addestrata sui dati as-is
    # mentre la fair pipeline verrà addestrata su dati sui vengono applicate strategie di fairness
    # volte a rimuovere discriminazione e bias nel dataset di training
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    # Strategia KFold
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    print(f'######### Training modelli #########')
    # fit del modello sul training set dell'i-esima iterazione
    lr_model_pipeline.fit(X_train,y_train.values.ravel())
    rf_model_pipeline.fit(X_train,y_train.values.ravel())
    svm_model_pipeline.fit(X_train,y_train.values.ravel())
    xgb_model_pipeline.fit(X_train,y_train.values.ravel())

    # Stampiamo metriche di valutazione per il modello
    print(f'######### Testing modelli #########')
    validate(lr_model_pipeline,'lr', X_test, y_test,True)
    validate(rf_model_pipeline,'rf',X_test,y_test)
    validate(svm_model_pipeline,'svm',X_test,y_test)
    validate(xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_credit_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_credit_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_credit_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')
            
def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    f1 = f1_score(y_true=y_test,y_pred=pred)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/std_models/credit_metrics_report.txt",open_type) as f:
        f.write(f'{model_type}\n')
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nF1 score: {round(f1,3)}\n')
        f.write('\n')

def print_time(time,index):

    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    
    with open('./reports/time_reports/std/credit_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione iniziale di load dataset
load_dataset()