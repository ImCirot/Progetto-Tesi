import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.sklearn.metrics import *
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():

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

    X = df[features]

    y = df[target]


    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    lr_model = pickle.load(open('./output_models/std_models/lr_credit_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_credit_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_credit_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_credit_model.sav','rb'))
    
    df_train, df_test, X_train,X_test,y_train,y_test = train_test_split(df,X,y,test_size=0.2,random_state=42)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['Target'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['Target'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['Target'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['Target'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df)
    rf_post_pred = test_fairness(df_test,rf_df)
    svm_post_pred = test_fairness(df_test,svm_df)
    xgb_post_pred = test_fairness(df_test,xgb_df)


    # Stampiamo metriche di valutazione per il modello
    print(f'######### Testing risultati #########')
    validate(lr_model,lr_post_pred['Target'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['Target'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['Target'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['Target'],'xgb',X_test,y_test)
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(model,fair_pred,model_type,X,y,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    accuracy = accuracy_score(y_pred=fair_pred,y_true=y)

    y_proba = model.predict_proba(X)[::,1]

    auc_score = roc_auc_score(y,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/postprocessing_models/credit_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')

def test_fairness(dataset,pred):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

     ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    # Attributi sensibili
    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_attribute_names,
    )

    aif360_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_attribute_names,
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex_A93': 1}]
    unprivileged_groups = [{'sex_A93': 0}]

    # Calcolo della metrica sul dataset originale
    metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,seed=42)

    # Bilanciamo il dataset
    eqoddspost.fit(dataset_true=aif360_dataset,dataset_pred=aif360_pred)
    dataset_transformed =  eqoddspost.predict(aif360_pred,threshold=0.8)
    # Ricalcoliamo la metrica
    metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(metric_original.mean_difference(),'Mean_difference value before', first_message=True)
    print_fairness_metrics(metric_original.disparate_impact(),'DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(metric_transformed.mean_difference(),'Mean_difference value after')
    print_fairness_metrics(metric_transformed.disparate_impact(),'DI value after')
    
    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(metric_original.num_positives(privileged=True),'Num. of positive instances of priv_group before')
    print_fairness_metrics(metric_original.num_positives(privileged=False),'Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(metric_transformed.num_positives(privileged=True),'Num. of positive instances of priv_group after')
    print_fairness_metrics(metric_transformed.num_positives(privileged=False),'Num. of positive instances of unpriv_group after')

    # otteniamo i nuovi pesi forniti dall'oggetto che mitigano i problemi di fairness
    post_pred = dataset_transformed.convert_to_dataframe()[0]

    return post_pred

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/credit_postrocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
load_dataset()