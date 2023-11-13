from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.inprocessing import MetaFairClassifier
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep
import warnings

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

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    protected_features = [
        'marital_divorced','marital_married','marital_single','education_primary','education_secondary','education_tertiary'
    ]

    X = df[feature_names]
    y = df['y']

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_bank_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_bank_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_bank_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_bank_model.sav','rb'))


    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_train = processing_fairness(df,X_train,y_train,protected_features)

    X_postop_train = processed_train[feature_names]
    y_postop_train = processed_train['y']

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

    print(f'######### Testing Fairness #########')
    X_test_df = X_test.copy(deep=True)
    X_test_df['y'] = y_test

    lr_inproc_pred = X_test.copy(deep=True)
    lr_inproc_pred['y'] = post_lr_model_pipeline.predict(X_test)

    rf_inproc_pred =  X_test.copy(deep=True)
    rf_inproc_pred['y'] = post_rf_model_pipeline.predict(X_test)

    svm_inproc_pred =  X_test.copy(deep=True)
    svm_inproc_pred['y'] = post_svm_model_pipeline.predict(X_test)

    xgb_inproc_pred =  X_test.copy(deep=True)
    xgb_inproc_pred['y'] = post_xgb_model_pipeline.predict(X_test)

    lr_pred = X_test.copy(deep=True)
    lr_pred['y'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test.copy(deep=True)
    rf_pred['y'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test.copy(deep=True)
    svm_pred['y'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test.copy(deep=True)
    xgb_pred['y'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,'lr')
    eq_odds_fair_report(X_test_df,rf_pred,'rf')
    eq_odds_fair_report(X_test_df,svm_pred,'svm')
    eq_odds_fair_report(X_test_df,xgb_pred,'xgb')

    eq_odds_fair_report(X_test_df,lr_inproc_pred,'lr_inprocessing')
    eq_odds_fair_report(X_test_df,rf_inproc_pred,'rf_inprocessing')
    eq_odds_fair_report(X_test_df,svm_inproc_pred,'svm_inprocessing')
    eq_odds_fair_report(X_test_df,xgb_inproc_pred,'xgb_inprocessing')


    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocess_models/lr_aif360_bank_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocess_models/rf_aif360_bank_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocess_models/svm_aif360_bank_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocess_models/xgb_aif360_bank_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def processing_fairness(dataset,X_set,y_set,protected_features):

    fair_classifier = MetaFairClassifier(type='sr',seed=42)

    train_dataset = pd.DataFrame(X_set)

    train_dataset['y'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=protected_features
    )

    privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    unprivileged_groups = [{'marital_divorced': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)
    
    print_metrics(metrics_og.mean_difference(),f'Marital Mean difference pre inprocessing',first_message=True)
    print_metrics(metrics_og.disparate_impact(),f'Marital DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Marital Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Marital DI post inprocessing')

    privileged_groups = [{'education_primary': 1}]
    unprivileged_groups = [{'education_primary': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_metrics(metrics_og.mean_difference(),f'Education Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Education DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Education Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Education DI post inprocessing')

    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train

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

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Marital Eq. Odds difference')

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

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Education Eq. Odds difference')
    

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
    with open(f'./reports/inprocessing_models/aif360/bank_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/aif360/bank_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/bank_inprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} elapsed time: {time} seconds.\n')

warnings.filterwarnings('ignore',category=RuntimeWarning)
load_dataset()