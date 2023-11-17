from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
import xgboost as xgb
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import pickle
from datetime import datetime
from sklearn.feature_selection import f_classif,SelectKBest
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
        training_testing_models(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'CODE_GENDER','AGE_CAT'
    ]

    sample_weights = test_fairness(dataset)

    fair_dataset = dataset.copy(deep=True)

    fair_dataset['weights'] = sample_weights

    feature_names = dataset.columns.tolist()

    feature_names.remove('TARGET')

    X = dataset[feature_names]
    y = dataset['TARGET']

    X_fair = fair_dataset[feature_names]
    y_fair = fair_dataset['TARGET']
    weights = fair_dataset['weights']

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_home_credit_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_home_credit_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_home_credit_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_home_credit_model.sav','rb'))

    # settiamo i nostri modelli sul dataset fair
    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',LogisticRegression(max_iter=1000))
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',RandomForestClassifier())
    ])

    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',LinearSVC(dual='auto',max_iter=1000))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic',random_state=42))
    ])


    selector = SelectKBest(score_func=f_classif,k=60)
    selector.fit(X,y)
    mask = selector.get_support(indices=True)
    X_selected = X.iloc[:,mask]
    X_selected['AGE_CAT'] = X['AGE_CAT']

    selector.fit(X_fair,y_fair)
    mask = selector.get_support(indices=True)
    X_fair_selected = X_fair.iloc[:,mask]
    X_fair_selected['AGE_CAT'] = X_fair['AGE_CAT']

    X_train, X_test, y_train, y_test = train_test_split(X_selected,y,test_size=0.2,random_state=42)
    X_fair_train, X_fair_test, y_fair_train, y_fair_test, weights_train, weights_test = train_test_split(X_fair_selected,y_fair,weights,test_size=0.2,random_state=42)


    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)

    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr', X_fair_test, y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    print('######### Testing Fairness #########')

    X_test_fair_df = X_fair_test.copy(deep=True)
    X_test_fair_df['TARGET'] = y_fair_test

    X_test_df = X_test.copy(deep=True)
    X_test_df['TARGET'] = y_test

    lr_fair_pred = X_test_df.copy(deep=True)
    lr_fair_pred['TARGET'] = lr_fair_model_pipeline.predict(X_test)

    rf_fair_pred =  X_test_df.copy(deep=True)
    rf_fair_pred['TARGET'] = rf_fair_model_pipeline.predict(X_test)

    svm_fair_pred =  X_test_df.copy(deep=True)
    svm_fair_pred['TARGET'] = svm_fair_model_pipeline.predict(X_test)

    xgb_fair_pred =  X_test_df.copy(deep=True)
    xgb_fair_pred['TARGET'] = xgb_fair_model_pipeline.predict(X_test)

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['TARGET'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test_df.copy(deep=True)
    rf_pred['TARGET'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test_df.copy(deep=True)
    svm_pred['TARGET'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test_df.copy(deep=True)
    xgb_pred['TARGET'] = xgb_model_pipeline.predict(X_test)

    std_predictions = {
        'lr_std':lr_pred,
        'rf_std': rf_pred,
        'svm_std': svm_pred,
        'xgb_std': xgb_pred,
    }

    fair_prediction = {
        'lr_fair': lr_fair_pred,
        'rf_fair':rf_fair_pred,
        'svm_fair': svm_fair_pred,
        'xgb_fair': xgb_fair_pred
    }


    for name,prediction in std_predictions.items():
        eq_odds_fair_report(X_test_df,prediction,name)

    for name,prediction in fair_prediction.items():
        eq_odds_fair_report(X_test_df,prediction,name)
    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_aif360_home_credit_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_aif360_home_credit_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_aif360_home_credit_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_aif360_home_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    pred = ml_model.predict(X_test)
    
    accuracy = accuracy_score(y_test,pred)

    f1 = f1_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    if first:
        open_type = "w"
    else:
        open_type = "a"
        
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/preprocessing_models/aif360/home_credit_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'F1 Score: {round(f1,3)}\n')
        f.write(f'Precision: {round(precision,3)}\n')
        f.write(f'Recall: {round(recall,3)}\n')
        f.write('\n')

def test_fairness(dataset):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    protected_features = [
        'AGE_CAT','CODE_GENDER'
    ]

    aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=protected_features
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva

    privileged_group = [{'CODE_GENDER': 1} | {'AGE_CAT':1}]
    unprivileged_group = [{'CODE_GENDER': 0,'AGE_CAT':0}]

    gender_privileged_group = [{'CODE_GENDER': 1}]
    gender_unprivileged_group = [{'CODE_GENDER': 0}]

    age_privileged_group = [{'AGE_CAT': 1}]
    age_unprivileged_group = [{'AGE_CAT': 0}]

    gender_metric_og = BinaryLabelDatasetMetric(dataset=aif_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    age_metric_og = BinaryLabelDatasetMetric(dataset=aif_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(gender_metric_og.mean_difference(),'Gender mean_difference before',first_message=True)
    print_fairness_metrics(gender_metric_og.disparate_impact(),"Gender DI before")

    print_fairness_metrics(age_metric_og.mean_difference(),'Age mean_difference before')
    print_fairness_metrics(age_metric_og.disparate_impact(),"Age DI before")
    
    gender_RW = Reweighing(unprivileged_groups=unprivileged_group,privileged_groups=privileged_group)

    trans_dataset = gender_RW.fit_transform(aif_dataset)

    gender_metric_trans = BinaryLabelDatasetMetric(dataset=trans_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    age_metric_trans = BinaryLabelDatasetMetric(dataset=trans_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(gender_metric_trans.mean_difference(),'Gender mean_difference after')
    print_fairness_metrics(gender_metric_trans.disparate_impact(),"Gender DI after")

    print_fairness_metrics(age_metric_trans.mean_difference(),'Age mean_difference after')
    print_fairness_metrics(age_metric_trans.disparate_impact(),"Age DI after")

    sample_weights = trans_dataset.instance_weights

    return sample_weights


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

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Gender Eq. Odds difference')

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

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model age Eq. Odds difference')

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/preprocessing/aif360/home_credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/home_credit_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()