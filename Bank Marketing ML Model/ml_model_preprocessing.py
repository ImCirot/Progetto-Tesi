from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_and_testing_models(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(300)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_and_testing_models(df):

    df_fair = df.copy(deep=True)

    instance_weights = test_fairness(df)
    
    df_fair['weights'] = instance_weights

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    protected_features = [
        'marital_divorced','marital_married','marital_single','education_primary','education_secondary','education_tertiary'
    ]

    X = df[feature_names]
    y = df['y']

    X_fair = df_fair[feature_names]
    y_fair = df_fair['y']
    sample_weights = df_fair['weights']

    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',LogisticRegression())
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',RandomForestClassifier())
    ])
    
    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',SVC(probability=True))
    ])
    
    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])


    X_fair_train, X_fair_test, y_fair_train, y_fair_test, sample_weights_train, sample_weights_test = train_test_split(X,y,sample_weights,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)

    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_aif360_bank_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_aif360_bank_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_aif360_bank_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_aif360_bank_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def test_fairness(dataset):
    maritial_features = [
        'marital_divorced','marital_married','marital_single'
    ]

    education_features = [
        'education_primary','education_secondary','education_tertiary'
    ]

    marital_df_aif360 = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=maritial_features
    )

    marital_privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    marital_unprivileged_groups = [{'marital_divorced': 1}]

    marital_metric_original = BinaryLabelDatasetMetric(dataset=marital_df_aif360,unprivileged_groups=marital_unprivileged_groups, privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_original.disparate_impact(),'(Marital) DI before', first_message=True)
    print_fairness_metrics(marital_metric_original.mean_difference(),'(Marital) mean_difference before')

    marital_RW = Reweighing(privileged_groups=marital_privileged_groups,unprivileged_groups=marital_unprivileged_groups)

    marital_df_trans = marital_RW.fit_transform(marital_df_aif360)

    marital_metric_trans =  BinaryLabelDatasetMetric(dataset=marital_df_trans,unprivileged_groups=marital_unprivileged_groups,privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_trans.disparate_impact(),'(Marital) DI after')
    print_fairness_metrics(marital_metric_trans.mean_difference(),'(Marital) mean_difference after')

    df_mod = marital_df_trans.convert_to_dataframe()[0]

    sample_weights = marital_df_trans.instance_weights

    df_mod['weights'] = sample_weights

    ed_df_aif360 = BinaryLabelDataset(
        df=df_mod,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
        instance_weights_name =['weights']
    )

    ed_privileged_groups = [{'education_primary': 1}]
    ed_unprivileged_groups = [{'education_primary': 0}]

    ed_metric_original = BinaryLabelDatasetMetric(dataset=ed_df_aif360,unprivileged_groups=ed_unprivileged_groups, privileged_groups=ed_privileged_groups)

    print_fairness_metrics(ed_metric_original.disparate_impact(),'(Ed.) DI before')
    print_fairness_metrics(ed_metric_original.mean_difference(),'(Ed.) Mean_difference before')

    ed_RW = Reweighing(privileged_groups=ed_privileged_groups,unprivileged_groups=ed_unprivileged_groups)

    ed_df_trans = ed_RW.fit_transform(dataset=ed_df_aif360)

    ed_metric_trans =  BinaryLabelDatasetMetric(dataset=ed_df_trans,unprivileged_groups=ed_unprivileged_groups,privileged_groups=ed_privileged_groups)

    print_fairness_metrics(ed_metric_trans.disparate_impact(),'(Ed.) DI after')
    print_fairness_metrics(ed_metric_trans.mean_difference(),'(Ed.) Mean_difference after')

    sample_weights = ed_df_trans.instance_weights

    return sample_weights


def validate(ml_model,ml_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"

    with  open(f"./reports/preprocessing_models/aif360/bank_metrics_report.txt",open_type) as f:
        f.write(f"{ml_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')


def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/preprocessing/aif360/bank_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/bank_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()