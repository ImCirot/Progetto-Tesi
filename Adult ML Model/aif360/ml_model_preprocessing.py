import numpy as np 
import pandas as pd 
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from codecarbon import track_emissions
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
import xgboost as xgb
from sklearn.svm import SVC
import pickle
import warnings
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione di load del dataset e drop features superflue

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # drop ID dal dataset
    df.drop('ID',inplace=True,axis=1)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(300)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione di apprendimento del modello sul dataset

    # setting variabili protette
    protected_features_names = [
        'race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male'
    ]

    fair_dataset = dataset.copy(deep=True)

    # creiamo un dataset fair effettuando modifiche al dataset originale
    sample_weights = test_fairness(dataset)

    fair_dataset['weights'] = sample_weights

    # setting nomi features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo il nome della feature target dalla lista nomi features
    features.remove('salary')

    # setting nome target feature
    target = ['salary']

    # setting dataset features
    X = dataset[features]
    X_fair = fair_dataset[features]

    # setting dataset target feature
    y = dataset[target]
    y_fair = fair_dataset[target]

    sample_weights = fair_dataset['weights']

    # costruiamo un modello tramite pipeline su cui utilizzare un dataset opportunamente modificato per aumentare fairness
    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',LogisticRegression())
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',RandomForestClassifier())
    ])

    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',SVC(probability=True))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic',random_state=42))
    ])

    X_fair_train, X_fair_test, y_fair_train, y_fair_test, sample_weights_train, sample_weights_test = train_test_split(X,y,sample_weights,test_size=0.2,random_state=42)

    # training del modello sul training set 
    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)

    # calcolo metriche di valutazione sul modello fair 
    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)


    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_aif360_adult_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_aif360_adult_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_aif360_adult_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_aif360_adult_model.sav','wb'))

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
    with  open(f"./reports/preprocessing_models/adult_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')

def test_fairness(original_dataset):
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
    
    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui "bianchi" come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di razze diverse, che vengono settati come appartenenti al gruppo sfavorito.
    race_privileged_groups = [{'race_White': 1}]
    race_unprivileged_groups = [{'race_White': 0}]

    # Calcolo della metrica sul dataset originale
    race_metric_original = BinaryLabelDatasetMetric(dataset=aif_race_dataset, unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)    
    
    # stampiamo la metrica mean_difference sul file di report    
    # (differenza fra predizioni positive di indivudi sfavoriti rispetto alle predizioni positive degli individui favoriti)
    print_fairness_metrics(race_metric_original.mean_difference(),'Race mean_difference before',first_message=True)
    print_fairness_metrics(race_metric_original.disparate_impact(),'Race DI before')
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    race_dataset_transformed = RACE_RW.fit_transform(aif_race_dataset)

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

    sample_weights = race_dataset_transformed.instance_weights

    new_dataset['weights'] = sample_weights

    # setting nome varibili sensibili legate al sesso
    sex_features = ['sex_Male','sex_Female']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "sex"
    aif_sex_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
        instance_weights_name=['weights']
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
    
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    SEX_RW = Reweighing(unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    sex_dataset_transformed = SEX_RW.fit_transform(aif_sex_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=sex_dataset_transformed,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(sex_metric_transformed.mean_difference(),'Sex mean_difference value after')
    print_fairness_metrics(sex_metric_transformed.disparate_impact(),'Sex DI value after')

    sample_weights = sex_dataset_transformed.instance_weights

    return sample_weights

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/preprocessing/adult_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/adult_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')


warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dataset()