from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
import xgboost as xgb
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing,EqOddsPostprocessing
import pickle
from sklearn.feature_selection import f_classif,SelectKBest
from datetime import datetime
from time import sleep


def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Home Credit Dataset/dataset.csv')

    # drop ID dal dataframe
    df.drop('ID', inplace=True,axis=1)

    # richiamo funzione di training e testing dei modelli
    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_testing_models(df,i)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(30)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_testing_models(dataset,index):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'CODE_GENDER','AGE_CAT'
    ]

    feature_names = dataset.columns.tolist()

    feature_names.remove('TARGET')

    X = dataset[feature_names]
    y = dataset['TARGET']

    lr_model = pickle.load(open('./output_models/std_models/lr_home_credit_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_home_credit_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_home_credit_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_home_credit_model.sav','rb'))
    

    selector = SelectKBest(score_func=f_classif,k=60)
    selector.fit(X,y)
    mask = selector.get_support(indices=True)
    X_selected = X.iloc[:,mask]
    X_selected['AGE_CAT'] = X['AGE_CAT']
    df_selected = pd.DataFrame(X_selected)
    df_selected['TARGET'] = dataset['TARGET']

    df_train,df_test,X_train, X_test, y_train, y_test = train_test_split(df_selected,X_selected,y,test_size=0.2,random_state=index)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['TARGET'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['TARGET'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['TARGET'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['TARGET'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df,'lr',True)
    rf_post_pred = test_fairness(df_test,rf_df,'rf')
    svm_post_pred = test_fairness(df_test,svm_df,'svm')
    xgb_post_pred = test_fairness(df_test,xgb_df,'xgb')

    target = ['TARGET']

    lr_post = df_test.copy(deep=True)
    lr_post[target] = lr_post_pred.values

    rf_post = df_test.copy(deep=True)
    rf_post[target] = lr_post_pred.values
    svm_post = df_test.copy(deep=True)
    svm_post[target] = lr_post_pred.values
    xgb_post = df_test.copy(deep=True)
    xgb_post[target] = lr_post_pred.values

    eq_odds_fair_report(df_test,lr_df,'lr')
    eq_odds_fair_report(df_test,rf_df,'rf')
    eq_odds_fair_report(df_test,svm_df,'svm')
    eq_odds_fair_report(df_test,xgb_df,'xgb')

    eq_odds_fair_report(df_test,lr_post,'lr_post')
    eq_odds_fair_report(df_test,rf_post,'rf_post')
    eq_odds_fair_report(df_test,svm_post,'svm_post')
    eq_odds_fair_report(df_test,xgb_post,'xgb_post')

    print(f'######### Testing Risultati #########')
    validate(lr_model,lr_post_pred['TARGET'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['TARGET'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['TARGET'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['TARGET'],'xgb',X_test,y_test)

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def eq_odds_fair_report(dataset,prediction,name):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['CODE_GENDER']
    )

    aif_gender_prediction = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['CODE_GENDER']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    gender_privileged_group = [{'CODE_GENDER': 1}]
    gender_unprivileged_group = [{'CODE_GENDER': 0}]

    metrics = ClassificationMetric(dataset=aif_gender_dataset,classified_dataset=aif_gender_prediction,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(metrics.equal_opportunity_difference(),f'{name}_model Gender Eq. Odds difference')

    aif_age_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['AGE_CAT']
    )

    aif_age_prediction = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['AGE_CAT']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    age_privileged_group = [{'AGE_CAT': 1}]
    age_unprivileged_group = [{'AGE_CAT': 0}]

    metrics = ClassificationMetric(dataset=aif_age_dataset,classified_dataset=aif_age_prediction,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(metrics.equal_opportunity_difference(),f'{name}_model age Eq. Odds difference')


def validate(model,fair_pred,model_type,X,y,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    accuracy = accuracy_score(y_pred=fair_pred,y_true=y)

    f1 = f1_score(y,fair_pred)

    precision = precision_score(y,fair_pred)

    recall = recall_score(y,fair_pred)  

    if first:
        open_type = "w"
    else:
        open_type = "a"
        
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/postprocessing_models/aif360/home_credit_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'F1 Score: {round(f1,3)}\n')
        f.write(f'Precision: {round(precision,3)}\n')
        f.write(f'Recall: {round(recall,3)}\n')
        f.write('\n')

def test_fairness(dataset,pred,name,first_message=False):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['CODE_GENDER']
    )

    aif_gender_pred= BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['CODE_GENDER']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    gender_privileged_group = [{'CODE_GENDER': 1}]
    gender_unprivileged_group = [{'CODE_GENDER': 0}]

    gender_metric_og = BinaryLabelDatasetMetric(dataset=aif_gender_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_og.mean_difference(),f'{name}_model Gender mean_difference before',first_message)
    print_fairness_metrics(gender_metric_og.disparate_impact(),f"{name}_model Gender DI before")
    
    eqoddspost = EqOddsPostprocessing(privileged_groups=gender_privileged_group, unprivileged_groups=gender_unprivileged_group,seed=42)

    gender_trans_dataset = eqoddspost.fit_predict(aif_gender_dataset,aif_gender_pred)

    gender_metric_trans = BinaryLabelDatasetMetric(dataset=gender_trans_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_trans.mean_difference(),f'{name}_model Gender mean_difference after')
    print_fairness_metrics(gender_metric_trans.disparate_impact(),f"{name}_model Gender DI after")

    new_dataset = gender_trans_dataset.convert_to_dataframe()[0]

    aif_age_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['AGE_CAT']
    )

    aif_age_pred= BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=['AGE_CAT']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    age_privileged_group = [{'AGE_CAT': 1}]
    age_unprivileged_group = [{'AGE_CAT': 0}]

    age_metric_og = BinaryLabelDatasetMetric(dataset=aif_age_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_og.mean_difference(),f'{name}_model age mean_difference before')
    print_fairness_metrics(age_metric_og.disparate_impact(),f"{name}_model age DI before")
    
    eqoddspost = EqOddsPostprocessing(privileged_groups=age_privileged_group, unprivileged_groups=age_unprivileged_group,seed=42)

    
    age_trans_dataset = eqoddspost.fit_predict(aif_age_dataset,aif_age_pred)

    age_metric_trans = BinaryLabelDatasetMetric(dataset=age_trans_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_trans.mean_difference(),f'{name}_model age mean_difference after')
    print_fairness_metrics(age_metric_trans.disparate_impact(),f"{name}_model age DI after")

    postop_pred = age_trans_dataset.convert_to_dataframe()[0]
    
    target = ['TARGET']
    return postop_pred[target]

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/aif360/home_credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/home_credit_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()