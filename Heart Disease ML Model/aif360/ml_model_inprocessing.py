import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():
    ##funzione di load del dataset

    df = pd.read_csv("./Heart Disease Dataset/dataset.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_and_testing_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(offline=True, country_iso_code="ITA")
def training_and_testing_model(df):
    ## Funzione per il training e testing del modello scelto
    features = df.columns.tolist()
    features.remove('num')

    target = ['num']

    X = df[features]

    y = df[target]

    protected_attribute_names = [
        'sex','age'
    ]

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_heart_disease_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_heart_disease_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_heart_disease_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_heart_disease_model.sav','rb'))

    post_lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(random_state=42))
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,random_state=42))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_train = processing_fairness(df,X_train,y_train,protected_attribute_names)

    X_postop_train = processed_train[features]
    y_postop_train = processed_train['num']

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
    X_test_df['num'] = y_test

    lr_inproc_pred = X_test.copy(deep=True)
    lr_inproc_pred['num'] = post_lr_model_pipeline.predict(X_test)

    rf_inproc_pred =  X_test.copy(deep=True)
    rf_inproc_pred['num'] = post_rf_model_pipeline.predict(X_test)

    svm_inproc_pred =  X_test.copy(deep=True)
    svm_inproc_pred['num'] = post_svm_model_pipeline.predict(X_test)

    xgb_inproc_pred =  X_test.copy(deep=True)
    xgb_inproc_pred['num'] = post_xgb_model_pipeline.predict(X_test)

    lr_pred = X_test.copy(deep=True)
    lr_pred['num'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test.copy(deep=True)
    rf_pred['num'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test.copy(deep=True)
    svm_pred['num'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test.copy(deep=True)
    xgb_pred['num'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,'lr')
    eq_odds_fair_report(X_test_df,rf_pred,'rf')
    eq_odds_fair_report(X_test_df,svm_pred,'svm')
    eq_odds_fair_report(X_test_df,xgb_pred,'xgb')

    eq_odds_fair_report(X_test_df,lr_inproc_pred,'lr_inprocessing')
    eq_odds_fair_report(X_test_df,rf_inproc_pred,'rf_inprocessing')
    eq_odds_fair_report(X_test_df,svm_inproc_pred,'svm_inprocessing')
    eq_odds_fair_report(X_test_df,xgb_inproc_pred,'xgb_inprocessing')


    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocessing_models/aif360/lr_heart_disease_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocessing_models/aif360/rf_heart_disease_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocessing_models/aif360/svm_heart_disease_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocessing_models/aif360/xgb_heart_disease_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def processing_fairness(dataset,X_set,y_set,protected_features):

    fair_classifier = MetaFairClassifier(tau=0,type='sr',seed=42)

    train_dataset = pd.DataFrame(X_set)
    train_dataset['num'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=protected_features,
    )

    sex_privileged_groups = [{'sex':1}]
    sex_unprivileged_groups = [{'sex':0}]

    age_privileged_groups = [{'age':1}]
    age_unprivileged_groups = [{'age':0}]

    sex_metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=sex_privileged_groups,unprivileged_groups=sex_unprivileged_groups)
    
    age_metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=age_privileged_groups,unprivileged_groups=age_unprivileged_groups)

    print_inproc_metrics(sex_metrics_og.mean_difference(),f'Gender Mean difference pre inprocessing',first_message=True)
    print_inproc_metrics(sex_metrics_og.disparate_impact(),f'Gender DI pre inprocessing')

    print_inproc_metrics(age_metrics_og.mean_difference(),f'Age Mean difference pre inprocessing')
    print_inproc_metrics(age_metrics_og.disparate_impact(),f'Age DI pre inprocessing')

    fair_df = fair_classifier.fit_predict(dataset=aif_train)

    sex_metrics_trans = BinaryLabelDatasetMetric(dataset=fair_df,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    age_metrics_trans = BinaryLabelDatasetMetric(dataset=fair_df,unprivileged_groups=age_unprivileged_groups,privileged_groups=age_privileged_groups)

    print_inproc_metrics(sex_metrics_trans.mean_difference(),f'Gender Mean difference post inprocessing')
    print_inproc_metrics(sex_metrics_trans.disparate_impact(),f'Gender DI post inprocessing')

    print_inproc_metrics(age_metrics_trans.mean_difference(),f'Age Mean difference post inprocessing')
    print_inproc_metrics(age_metrics_trans.disparate_impact(),f'Age DI post inprocessing')

    df_train = fair_df.convert_to_dataframe()[0]

    return df_train

def eq_odds_fair_report(dataset,prediction,name):
    # Attributi sensibili
    protected_attribute_names = [
        'age','sex'
    ]

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=protected_attribute_names,
    )

    aif360_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=protected_attribute_names,
    )


    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    metrics = ClassificationMetric(dataset=aif360_dataset,classified_dataset=aif360_pred,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_inproc_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Gender Eq. Odds difference')

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['age'],
    )

    aif360_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['age'],
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    metrics = ClassificationMetric(dataset=aif360_dataset,classified_dataset=aif360_pred,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_inproc_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Age Eq. Odds difference')

def print_inproc_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'

    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/aif360/heart_disease_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def validate(ml_model,model_type,X_test,y_test,first=False):
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
    with  open(f'./reports/inprocessing_models/aif360/heart_disease_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nF1 score: {round(f1,3)}\n')
        f.write(f"Precision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/heart_disease_inprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
load_dataset()
