import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns
import pickle
from codecarbon import track_emissions
from datetime import datetime

def training_model(dataset):
    ## funzione che addestra il modello sul dataset utilizzando strategia KFold

    # evidenziamo le features utili alla predizione
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista features la feature target
    features.remove('Target')

    # evidenziamo gli attributi sensibili del dataset
    sex_features = [
        'sex_A91','sex_A92','sex_A93','sex_A94'
    ]

    # settiamo la nostra X sulle sole variabili di features
    X = dataset[features]

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    dataset['Target'] = dataset['Target'].replace(2,0)

    # settiamo la nostra y sulla variabile da predire
    y = dataset['Target']

    # settiamo un dataframe contenente solamente i valori degli attributi sensibili (utile per utilizzare il framework FairLearn)
    g = dataset[sex_features]

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_credit_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_credit_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_credit_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_credit_model.sav','rb'))

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

    X_train, X_test, y_train, y_test,g_train,g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)

    # modifichiamo i modelli con postop di fairness

    print(f'######### Training modelli #########')
    lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
    rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
    svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
    xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

    # validiamo i nuovi modelli 
    print(f'######### Testing modelli #########')
    validate(lr_threshold,'lr',X_test,y_test,g_test,True)
    validate(rf_threshold,'rf',X_test,y_test,g_test)
    validate(svm_threshold,'svm',X_test,y_test,g_test)
    validate(xgb_threshold,'xgb',X_test,y_test,g_test)

    # linea di codice per plottare il accuracy e selection_rate del modello con operazione di postop
    # plot_threshold_optimizer(lr_threshold)
    # plot_threshold_optimizer(rf_threshold)
    # plot_threshold_optimizer(svm_threshold)
    # plot_threshold_optimizer(xgb_threshold)

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

        sex_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/inprocessing/fairlearn/credit_report.txt',open_type) as f:
            f.write(f'{name}_sex DI: {round(sex_DI,3)}\n')

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_threshold,open('./output_models/inprocess_models/threshold_lr_fairlearn_credit_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/inprocess_models/threshold_rf_fairlearn_credit_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/inprocess_models/threshold_svm_fairlearn_credit_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/inprocess_models/threshold_xgb_fairlearn_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,g_test,first=False):
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
    with  open(f'./reports/inprocessing_models/fairlearn/credit_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./German Credit Dataset/dataset_modificato.csv')

    df.drop('ID', axis=1, inplace=True)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        training_model(df)

def print_time(time):
    with open('./reports/time_reports/fairlearn/credit_inprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)