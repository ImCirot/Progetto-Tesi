import pandas as pd
import numpy as np
from fairlearn.reductions import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from fairlearn.metrics import MetricFrame,demographic_parity_difference,demographic_parity_ratio,equalized_odds_difference
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from fairlearn.postprocessing import ThresholdOptimizer,plot_threshold_optimizer
import pickle
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

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
    ## funzione di sviluppo del modello

    # drop delle features superflue
    dataset = dataset.drop("ID",axis=1)

    # lista con tutte le features del dataset
    features = dataset.columns.tolist()

    # drop dalla lista del nome della variabile target
    features.remove('salary')

    # setting lista contenente nomi degli attributi protetti
    protected_features_names = ['age','sex_Female','sex_Male']
    sex_features_names = ['sex_Female','sex_Male']
    age_feature = ['age']

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente valori attributi sensibili
    g= dataset[protected_features_names]

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_adult_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_adult_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_adult_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_adult_model.sav','rb'))

    # costruiamo un operatore di postprocessing per cercare di ottimizzare il modello 
    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='auto',
        prefit=True
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='auto',
        prefit=True
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='auto',
        prefit=True
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='auto',
        prefit=True
    )

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
    rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
    svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
    xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

    print(f'######### Testing modelli #########')
    validate(lr_threshold,"lr",X_test,y_test,g_test,True)
    validate(rf_threshold,'rf',X_test,y_test,g_test)
    validate(svm_threshold,'svm',X_test,y_test,g_test)
    validate(xgb_threshold,'xgb',X_test,y_test,g_test)

    # linea di codice per plottare il accuracy e selection_rate del modello con operazione di postop
    # plot_threshold_optimizer(lr_threshold)
    # plot_threshold_optimizer(rf_threshold)
    # plot_threshold_optimizer(svm_threshold)
    # plot_threshold_optimizer(xgb_threshold)
    
    # per stampare i grafici generati
    # plt.show()

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

        sex_DI_value = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])
        sex_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])
        sex_eq_odds_diff = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])

        age_DI_value = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_eq_odds_diff = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/postprocessing/fairlearn/adult_report.txt',open_type) as f:
            f.write(f'{name} sex DI: {round(sex_DI_value,3)}\n')
            f.write(f'{name} sex mean_diff: {round(sex_mean_diff,3)}\n')
            f.write(f'{name} sex eq_odds_diff: {round(sex_eq_odds_diff,3)}\n')
            f.write(f'{name} age DI: {round(age_DI_value,3)}\n')
            f.write(f'{name} age mean_diff: {round(age_mean_diff,3)}\n')
            f.write(f'{name} age eq_odds_diff: {round(age_eq_odds_diff,3)}\n')

    
    # salviamo i modelli ottenuti
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_threshold,open('./output_models/postprocessing_models/fairlearn/threshold_lr_adult_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/postprocessing_models/fairlearn/threshold_rf_adult_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/postprocessing_models/fairlearn/threshold_svm_adult_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/postprocessing_models/fairlearn/threshold_xgb_adult_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model, model_type, X_test, y_test, g_test, first=False):
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
    with open(f"./reports/postprocessing_models/fairlearn/adult_metrics_report.txt",open_type) as f:
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

    with open('./reports/time_reports/fairlearn/adult_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()