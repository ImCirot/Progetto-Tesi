import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.model_selection import train_test_split
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns
import pickle
from codecarbon import track_emissions
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./Heart Disease Dataset/dataset.csv')

    df.drop('ID', axis=1, inplace=True)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione che addestra e valuta dei modelli con inprocessing sul dataset originale
    
    # otteniamo la lista delle features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista la feature target
    features.remove("num")

    # settiamo il nome delle features sensibili
    protected_features = [
        'age','sex'
    ]

    sex_feature = ['sex']
    age_feature = ['age']

    # setting delle variabili X,y e la variabile g degli attributi sensibili
    X = dataset[features]
    y = dataset['num']
    g = dataset[protected_features]

    # carichiamo i modelli standard per effettuare dei confronti di fairness post training e testing dei modelli fair
    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_heart_disease_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_heart_disease_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_heart_disease_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_heart_disease_model.sav','rb'))

    # creiamo i modelli che verranno addestrati tramite inprocessing sul dataset originale partendo dal modello standard
    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True,
        objective='accuracy_score'
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True,
        objective='accuracy_score'
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True,
        objective='accuracy_score'
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True,
        objective='accuracy_score'
    )

    # setting subset train/test X e y
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)

    # addestriamo i modelli
    print(f'######### Training modelli #########')
    lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
    rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
    svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
    xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

    # validiamo i modelli
    print(f'######### Testing modelli #########')
    validate(lr_threshold,'lr',X_test,y_test,g_test,True)
    validate(rf_threshold,'rf',X_test,y_test,g_test)
    validate(svm_threshold,'svm',X_test,y_test,g_test)
    validate(xgb_threshold,'xgb',X_test,y_test,g_test)

    print(f'######### Testing Fairness #########')
    lr_std_pred = lr_model_pipeline.predict(X_test)
    lr_threshold_pred = lr_threshold.predict(X_test,sensitive_features=g_test)

    rf_std_pred = rf_model_pipeline.predict(X_test)
    rf_threshold_pred = rf_threshold.predict(X_test,sensitive_features=g_test)

    svm_std_pred = svm_model_pipeline.predict(X_test)
    svm_threshold_pred = svm_threshold.predict(X_test,sensitive_features=g_test)

    xgb_std_pred = xgb_model_pipeline.predict(X_test)
    xgb_threshold_pred = xgb_threshold.predict(X_test,sensitive_features=g_test)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_threshold': lr_threshold_pred,
        'rf_std': rf_std_pred,
        'rf_threshold':rf_threshold_pred,
        'svm_std': svm_std_pred,
        'svm_threshold': svm_threshold_pred,
        'xgb_std': xgb_std_pred,
        'xgb_threshold': xgb_threshold_pred
    }

    start = True

    for name,prediction in predictions.items():

        sex_DI_score = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        sex_eqodds = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        sex_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        age_DI_score = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_eqodds = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])


        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/postprocessing/fairlearn/heart_disease_report.txt',open_type) as f:
            f.write(f'{name} Sex DI: {round(sex_DI_score,3)}\n')
            f.write(f'{name} Sex mean_diff: {round(sex_mean_diff,3)}\n')
            f.write(f'{name} Sex eq_odds_diff: {round(sex_eqodds,3)}\n')
            f.write(f'{name} Age DI: {round(age_DI_score,3)}\n')
            f.write(f'{name} Age mean_diff: {round(age_mean_diff,3)}\n')
            f.write(f'{name} Age eq_odds_diff: {round(age_eqodds,3)}\n')


    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_threshold,open('./output_models/postprocessing_models/threshold_lr_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/postprocessing_models/threshold_rf_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/postprocessing_models/threshold_svm_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/postprocessing_models/threshold_xgb_fairlearn_heart_disease_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,g_test,first=False):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test,sensitive_features=g_test)

    accuracy = accuracy_score(y_test, pred)

    f1 = f1_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f'./reports/postprocessing_models/fairlearn/heart_disease_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nF1 score: {round(f1,3)}\n')
        f.write(f"Precision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/fairlearn/heart_disease_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()