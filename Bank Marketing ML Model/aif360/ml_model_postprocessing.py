from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    for i in range(1):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_and_testing_models(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(1)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_and_testing_models(df):
    feature_names = df.columns.tolist()

    feature_names.remove('y')

    protected_features = [
        'marital_divorced','marital_married','marital_single','education_primary','education_secondary','education_tertiary'
    ]

    X = df[feature_names]
    y = df['y']

    lr_model = pickle.load(open('./output_models/std_models/lr_bank_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_bank_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_bank_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_bank_model.sav','rb'))

    df_train, df_test, X_train,X_test,y_train,y_test = train_test_split(df,X,y,test_size=0.2,random_state=42)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['y'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['y'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['y'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['y'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df,'lr',True)
    rf_post_pred = test_fairness(df_test,rf_df,'rf')
    svm_post_pred = test_fairness(df_test,svm_df,'svm')
    xgb_post_pred = test_fairness(df_test,xgb_df,'xgb')

    eq_odds_fair_report(df_test,lr_df,'lr')
    eq_odds_fair_report(df_test,rf_df,'rf')
    eq_odds_fair_report(df_test,svm_df,'svm')
    eq_odds_fair_report(df_test,xgb_df,'xgb')

    eq_odds_fair_report(df_test,lr_post_pred,'lr_post')
    eq_odds_fair_report(df_test,rf_post_pred,'rf_post')
    eq_odds_fair_report(df_test,svm_post_pred,'svm_post')
    eq_odds_fair_report(df_test,xgb_post_pred,'xgb_post')


    print(f'######### Testing risultati #########')
    validate(lr_model,lr_post_pred['y'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['y'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['y'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['y'],'xgb',X_test,y_test)

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def test_fairness(dataset,pred,name,first_message=False):
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

    marital_pred_aif360 = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=maritial_features
    )


    marital_privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    marital_unprivileged_groups = [{'marital_divorced': 1}]

    marital_metric_original = BinaryLabelDatasetMetric(dataset=marital_df_aif360,unprivileged_groups=marital_unprivileged_groups, privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_original.disparate_impact(),f'{name}_model Marital DI before', first_message)
    print_fairness_metrics(marital_metric_original.mean_difference(),f'{name}_model Marital mean_difference before')

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=marital_privileged_groups, unprivileged_groups=marital_unprivileged_groups,seed=42)

    marital_df_trans = eqoddspost.fit_predict(marital_df_aif360,marital_pred_aif360,threshold=0.8)

    marital_metric_trans =  BinaryLabelDatasetMetric(dataset=marital_df_trans,unprivileged_groups=marital_unprivileged_groups,privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_trans.disparate_impact(),f'{name}_model Marital DI after')
    print_fairness_metrics(marital_metric_trans.mean_difference(),f'{name}_model Marital mean_difference after')

    df_mod = marital_df_trans.convert_to_dataframe()[0]
    
    ed_df_aif360 = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
    )

    ed_pred_aif360 = BinaryLabelDataset(
        df=df_mod,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
    )

    ed_privileged_groups = [{'education_primary': 1}]
    ed_unprivileged_groups = [{'education_primary': 0}]

    ed_metric_original = BinaryLabelDatasetMetric(dataset=ed_df_aif360,unprivileged_groups=ed_unprivileged_groups, privileged_groups=ed_privileged_groups)

    print_fairness_metrics(ed_metric_original.disparate_impact(),f'{name}_model Ed. DI before')
    print_fairness_metrics(ed_metric_original.mean_difference(),f'{name}_model Ed. Mean_difference before')

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=ed_unprivileged_groups, unprivileged_groups=ed_privileged_groups,seed=42)

    ed_df_trans = eqoddspost.fit_predict(ed_df_aif360,ed_pred_aif360,threshold=0.8)

    ed_metric_trans =  BinaryLabelDatasetMetric(dataset=ed_df_trans,unprivileged_groups=ed_privileged_groups,privileged_groups=ed_unprivileged_groups)

    print_fairness_metrics(ed_metric_trans.disparate_impact(),f'{name}_model Ed. DI after')
    print_fairness_metrics(ed_metric_trans.mean_difference(),f'{name}_model Ed. Mean_difference after')

    postop_dataset = ed_df_trans.convert_to_dataframe()[0]

    return postop_dataset

def eq_odds_fair_report(dataset,prediction,name):
    # Attributi sensibili
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

    marital_pred_aif360 = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=maritial_features
    )

    marital_privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    marital_unprivileged_groups = [{'marital_divorced': 1}]

    metrics = ClassificationMetric(dataset=marital_df_aif360,classified_dataset=marital_pred_aif360,privileged_groups=marital_privileged_groups,unprivileged_groups=marital_unprivileged_groups)

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Marital Eq. Odds difference')

    ed_df_aif360 = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
    )

    ed_pred_aif360 = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
    )

    ed_privileged_groups = [{'education_primary': 1}]
    ed_unprivileged_groups = [{'education_primary': 0}]

    metrics = ClassificationMetric(dataset=ed_df_aif360,classified_dataset=ed_pred_aif360,privileged_groups=ed_privileged_groups,unprivileged_groups=ed_unprivileged_groups)

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Education Eq. Odds difference')
    

def validate(model,fair_pred,model_type,X,y,first=False):

    accuracy = accuracy_score(y_pred=fair_pred,y_true=y)

    y_proba = model.predict_proba(X)[::,1]

    auc_score = roc_auc_score(y,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"

    with  open(f"./reports/postprocessing_models/aif360/bank_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
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
    with open(f"./reports/fairness_reports/postprocessing/aif360/bank_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/bank_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()