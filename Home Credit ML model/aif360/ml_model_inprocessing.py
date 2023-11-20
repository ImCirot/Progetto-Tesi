from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
from sklearn.feature_selection import f_classif,SelectKBest
import xgboost as xgb
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.inprocessing import *
from fairlearn.reductions import DemographicParity
import pickle
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

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_home_credit_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_home_credit_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_home_credit_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_home_credit_model.sav','rb'))

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),LinearSVC(dual='auto',max_iter=1000))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    selector = SelectKBest(score_func=f_classif,k=60)
    selector.fit(X,y)
    mask = selector.get_support(indices=True)
    X_selected = X.iloc[:,mask]
    X_selected['AGE_CAT'] = X['AGE_CAT']

    X_train, X_test, y_train, y_test = train_test_split(X_selected,y,test_size=0.2,random_state=index)

    processed_train = processing_fairness(dataset,X_train,y_train,sensible_features_names)

    X_postop_train = processed_train[X_selected.columns.tolist()]
    y_postop_train = y_train

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
    X_test_df['TARGET'] = y_test

    lr_inproc_pred = X_test.copy(deep=True)
    lr_inproc_pred['TARGET'] = post_lr_model_pipeline.predict(X_test)

    rf_inproc_pred =  X_test.copy(deep=True)
    rf_inproc_pred['TARGET'] = post_rf_model_pipeline.predict(X_test)

    svm_inproc_pred =  X_test.copy(deep=True)
    svm_inproc_pred['TARGET'] = post_svm_model_pipeline.predict(X_test)

    xgb_inproc_pred =  X_test.copy(deep=True)
    xgb_inproc_pred['TARGET'] = post_xgb_model_pipeline.predict(X_test)

    lr_pred = X_test.copy(deep=True)
    lr_pred['TARGET'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test.copy(deep=True)
    rf_pred['TARGET'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test.copy(deep=True)
    svm_pred['TARGET'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test.copy(deep=True)
    xgb_pred['TARGET'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,'lr')
    eq_odds_fair_report(X_test_df,rf_pred,'rf')
    eq_odds_fair_report(X_test_df,svm_pred,'svm')
    eq_odds_fair_report(X_test_df,xgb_pred,'xgb')

    eq_odds_fair_report(X_test_df,lr_inproc_pred,'lr_inprocessing')
    eq_odds_fair_report(X_test_df,rf_inproc_pred,'rf_inprocessing')
    eq_odds_fair_report(X_test_df,svm_inproc_pred,'svm_inprocessing')
    eq_odds_fair_report(X_test_df,xgb_inproc_pred,'xgb_inprocessing')

    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocessing_models/aif360/lr_home_credit_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocessing_models/aif360/rf_home_credit_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocessing_models/aif360/svm_home_credit_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocessing_models/aif360/xgb_home_credit_model.sav','wb'))

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

    print_metrics(metrics.equal_opportunity_difference(),f'{name}_model Gender Eq. Odds difference')

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

    print_metrics(metrics.equal_opportunity_difference(),f'{name}_model age Eq. Odds difference')

def processing_fairness(dataset,X_set,y_set,protected_features):

    fair_classifier = MetaFairClassifier(type='sr',tau=0.5,seed=42)

    train_dataset = pd.DataFrame(X_set)

    train_dataset['TARGET'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['TARGET'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'CODE_GENDER': 1}]
    unprivileged_groups = [{'CODE_GENDER': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)
    
    print_metrics(metrics_og.mean_difference(),f'Gender Mean difference pre inprocessing',first_message=True)
    print_metrics(metrics_og.disparate_impact(),f'Gender DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Gender Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Gender DI post inprocessing')

    privileged_groups = [{'AGE_CAT': 1}]
    unprivileged_groups = [{'AGE_CAT': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)
    
    print_metrics(metrics_og.mean_difference(),f'Age Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Age DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Age Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Age DI post inprocessing')

    
    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train

def validate(ml_model,model_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    f1 = f1_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    if first:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/inprocessing_models/aif360/home_credit_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'F1 Score: {round(f1,3)}\n')
        f.write(f'Precision: {round(precision,3)}\n')
        f.write(f'Recall: {round(recall,3)}\n')
        f.write('\n')

def print_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/aif360/home_credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/home_credit_inprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()