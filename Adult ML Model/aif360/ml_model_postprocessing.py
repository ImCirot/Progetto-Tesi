import numpy as np 
import pandas as pd 
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from codecarbon import track_emissions
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import pickle
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione di load del dataset e drop features superflue

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # drop ID dal dataset
    df.drop('ID',inplace=True,axis=1)

    training_model(df)

    # for i in range(10):
    #     print(f'########################### {i+1} esecuzione ###########################')
    #     start = datetime.now()
    #     training_model(df)
    #     end = datetime.now()
    #     elapsed = (end - start).total_seconds()
    #     print_time(elapsed,i)
    #     if(i < 9):
    #         print('########################### IDLE TIME START ###########################')
    #         sleep(300)
    #         print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione di apprendimento del modello sul dataset

    # setting variabili protette
    protected_features_names = [
        'race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male'
    ]

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

    lr_model = pickle.load(open('./output_models/std_models/lr_adult_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_adult_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_adult_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_adult_model.sav','rb'))

    df_train, df_test, X_train,X_test,y_train,y_test = train_test_split(dataset,X,y,test_size=0.2,random_state=42)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['salary'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['salary'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['salary'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['salary'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df)
    rf_post_pred = test_fairness(df_test,rf_df)
    svm_post_pred = test_fairness(df_test,svm_df)
    xgb_post_pred = test_fairness(df_test,xgb_df)

    print(f'######### Testing risultati #########')
    validate(lr_model,lr_post_pred['salary'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['salary'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['salary'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['salary'],'xgb',X_test,y_test)

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
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/postprocessing_models/adult_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')

def test_fairness(original_dataset,pred):
    ## funzione che testa la fairness del dataset tramite libreria AIF360 e restituisce un dataset fair opportunamente modificato

    race_features = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "race"
    aif_race_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_features,
        privileged_protected_attributes=['race_White']
    )
    
    aif_race_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_features,
        privileged_protected_attributes=['race_White']
    )

    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui "bianchi" come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di razze diverse, che vengono settati come appartenenti al gruppo sfavorito.
    race_privileged_groups = [{'race_White': 1}]
    race_unprivileged_groups = [{'race_White': 0}]

    # Calcolo della metrica sul dataset originale
    race_metric_original = BinaryLabelDatasetMetric(dataset=aif_race_dataset, unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)    
    
    print_fairness_metrics(race_metric_original.mean_difference(),'Race mean_difference before',first_message=True)
    print_fairness_metrics(race_metric_original.disparate_impact(),'Race DI before')

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='fnr',privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups,seed=42)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    race_dataset_transformed = eqoddspost.fit_predict(aif_race_dataset,aif_race_pred,threshold=0.8)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_dataset_transformed,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(race_metric_transformed.mean_difference(),'Race mean_difference after')
    print_fairness_metrics(race_metric_transformed.disparate_impact(),'Race DI after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(race_metric_original.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group before')
    print_fairness_metrics(race_metric_original.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(race_metric_transformed.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group after')
    print_fairness_metrics(race_metric_transformed.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group after')
    
    # passiamo ora a valutare il dataset sulla base della feature legata al genere

    new_dataset = race_dataset_transformed.convert_to_dataframe()[0]

    # setting nome varibili sensibili legate al sesso
    sex_features = ['sex_Male','sex_Female']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "sex"
    aif_sex_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    aif_sex_pred = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui di sesso maschile come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di sesso femminile, che vengono settati come appartenenti al gruppo sfavorito.
    sex_privileged_groups = [{'sex_Male': 1}]
    sex_unprivileged_groups = [{'sex_Female': 1}]

    # Calcolo della metrica sul dataset originale
    sex_metric_original = BinaryLabelDatasetMetric(dataset=aif_sex_dataset, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups) 
    
    # stampiamo la metrica mean_difference sul file di report    
    print_fairness_metrics(sex_metric_original.mean_difference(),'Sex mean_difference before')
    print_fairness_metrics(sex_metric_original.disparate_impact(),'Sex DI before')
    
    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='fnr',privileged_groups=sex_privileged_groups, unprivileged_groups=sex_unprivileged_groups,seed=42)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    sex_dataset_transformed = eqoddspost.fit_predict(aif_sex_dataset,aif_sex_pred,threshold=0.8)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=sex_dataset_transformed,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(sex_metric_transformed.mean_difference(),'Sex mean_difference value after')
    print_fairness_metrics(sex_metric_transformed.disparate_impact(),'Sex DI value after')

    new_dataset = sex_dataset_transformed.convert_to_dataframe()[0]

    return new_dataset

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/adult_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/adult_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()