import pandas as pd
import numpy as np
from fairlearn.reductions import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from fairlearn.metrics import MetricFrame,demographic_parity_difference,demographic_parity_ratio
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from fairlearn.postprocessing import ThresholdOptimizer,plot_threshold_optimizer
import pickle
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        training_model(df)

def training_model(dataset):
    ## funzione di sviluppo del modello

    # drop delle features superflue
    dataset.drop("ID",axis=1,inplace=True)

    # lista con tutte le features del dataset
    features = dataset.columns.tolist()

    # drop dalla lista del nome della variabile target
    features.remove('salary')

    # setting lista contenente nomi degli attributi protetti
    protected_features_names = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male']
    sex_features_names = ['sex_Female','sex_Male']
    race_features_names = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White']

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente valori attributi sensibili
    g= dataset[protected_features_names]
    g_sex = dataset[sex_features_names]
    g_race = dataset[race_features_names]

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_adult_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_adult_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_adult_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_adult_model.sav','rb'))

    # costruiamo un operatore di postprocessing per cercare di ottimizzare il modello 
    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
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
    lr_std_pred = lr_model_pipeline.predict(X)
    lr_threshold_pred = lr_threshold.predict(X,sensitive_features=g)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_threshold_pred = rf_threshold.predict(X,sensitive_features=g)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_threshold_pred = svm_threshold.predict(X,sensitive_features=g)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_threshold_pred = xgb_threshold.predict(X,sensitive_features=g)

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

        sex_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g_sex)
        race_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g_race)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/inprocessing/fairlearn/adult_report.txt',open_type) as f:
            f.write(f'{name}_sex DI: {round(sex_DI,3)}\n')
            f.write(f'{name}_race DI: {round(race_DI,3)}\n')

    
    # salviamo i modelli ottenuti
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_threshold,open('./output_models/inprocess_models/threshold_lr_fairlearn_adult_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/inprocess_models/threshold_rf_fairlearn_adult_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/inprocess_models/threshold_svm_fairlearn_adult_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/inprocess_models/threshold_xgb_fairlearn_adult_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model, model_type, X_test, y_test, g_test, first=False):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test,sensitive_features=g_test)

    accuracy = accuracy_score(y_test, pred)

    y_proba = ml_model.estimator.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    
    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/inprocessing_models/fairlearn/adult_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/fairlearn/adult_inprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')


start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)