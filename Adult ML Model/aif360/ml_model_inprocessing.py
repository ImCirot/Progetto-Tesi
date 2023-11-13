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

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression(max_iter=200))
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_train = processing_fairness(dataset,X_train,y_train,protected_features_names)

    X_postop_train = processed_train[features]
    y_postop_train = processed_train['salary']

    print(f'######### Training modelli #########')
    post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
    post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
    post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
    post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

    print(f'######### Testing modelli #########')
    validate(post_lr_model_pipeline,'lr',X_test,y_test,True)
    validate(post_rf_model_pipeline,'rf',X_test,y_test)
    validate(post_svm_model_pipeline,'svm',X_test,y_test)
    validate(post_xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocess_models/lr_aif360_adult_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocess_models/rf_aif360_adult_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocess_models/svm_aif360_adult_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocess_models/xgb_aif360_adult_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def processing_fairness(dataset,X_set,y_set,protected_features):

    fair_classifier = MetaFairClassifier(type='fdr',seed=0)

    train_dataset = pd.DataFrame(X_set)

    train_dataset['salary'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'race_White': 1}]
    unprivileged_groups = [{'race_White': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)
    
    print_metrics(metrics_og.mean_difference(),f'Race Mean difference pre inprocessing',first_message=True)
    print_metrics(metrics_og.disparate_impact(),f'Race DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Race Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Race DI post inprocessing')

    privileged_groups = [{'sex_Male': 1}]
    unprivileged_groups = [{'sex_Female': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_metrics(metrics_og.mean_difference(),f'Gender Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Gender DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Gender Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Gender DI post inprocessing')

    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train
    

def print_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/adult_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def validate(ml_model,model_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/inprocessing_models/adult_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/adult_inprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dataset()
