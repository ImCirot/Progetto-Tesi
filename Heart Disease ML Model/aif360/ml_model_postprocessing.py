import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing,EqOddsPostprocessing
from aif360.sklearn.metrics import *
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():

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
            sleep(1)
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

    lr_model = pickle.load(open('./output_models/std_models/lr_heart_disease_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_heart_disease_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_heart_disease_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_heart_disease_model.sav','rb'))
    
    df_train, df_test, X_train,X_test,y_train,y_test = train_test_split(df,X,y,test_size=0.2,random_state=42)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['num'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['num'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['num'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['num'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df,'lr',True)
    rf_post_pred = test_fairness(df_test,rf_df,'rf')
    svm_post_pred = test_fairness(df_test,svm_df,'svm')
    xgb_post_pred = test_fairness(df_test,xgb_df,'xgb')

    test_eqodds(df_test,lr_df,'lr')
    test_eqodds(df_test,rf_df,'rf')
    test_eqodds(df_test,svm_df,'svm')
    test_eqodds(df_test,xgb_df,'xgb')

    test_eqodds(df_test,lr_post_pred,'lr_post')
    test_eqodds(df_test,rf_post_pred,'rf_post')
    test_eqodds(df_test,svm_post_pred,'svm_post')
    test_eqodds(df_test,xgb_post_pred,'xgb_post')

    # Stampiamo metriche di valutazione per il modello
    print(f'######### Testing risultati #########')
    validate(lr_model,lr_post_pred['num'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['num'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['num'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['num'],'xgb',X_test,y_test)
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(model,fair_pred,model_type,X,y,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    accuracy = accuracy_score(y_pred=fair_pred,y_true=y)

    f1 = f1_score(y_pred=fair_pred,y_true=y)

    precision = precision_score(y_pred=fair_pred,y_true=y)

    recall = recall_score(y_pred=fair_pred,y_true=y)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/postprocessing_models/aif360/heart_dheart_disease_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'F1 Score: {round(f1,3)}\n')
        f.write(f"Precision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')

def test_fairness(dataset,pred,name,first_message=False):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['sex'],
    )

    aif360_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['sex'],
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex': 1}]
    unprivileged_groups =  [{'sex': 0}]

    # Calcolo della metrica sul dataset originale
    metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    
    caleqodds = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups,cost_constraint='weighted',seed=42)

    dataset_transformed = caleqodds.fit_predict(dataset_true=aif360_dataset,dataset_pred=aif360_pred,threshold=0.8)

    # Ricalcoliamo la metrica
    metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(metric_original.mean_difference(),f'{name}_model sex Mean_difference value before', first_message)
    print_fairness_metrics(metric_original.disparate_impact(),f'{name}_model sex DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(metric_transformed.mean_difference(),f'{name}_model sex Mean_difference value after')
    print_fairness_metrics(metric_transformed.disparate_impact(),f'{name}_model sex DI value after')

    # otteniamo i nuovi pesi forniti dall'oggetto che mitigano i problemi di fairness
    post_pred = dataset_transformed.convert_to_dataframe()[0]


    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['age'],
    )

    aif360_pred = BinaryLabelDataset(
        df=post_pred,
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

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups,seed=42)
    # Bilanciamo il dataset
    dataset_transformed =  eqoddspost.fit_predict(dataset_true=aif360_dataset,dataset_pred=aif360_pred,threshold=0.8)
    # Ricalcoliamo la metrica
    metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(metric_original.mean_difference(),f'{name}_model age Mean_difference value before', first_message)
    print_fairness_metrics(metric_original.disparate_impact(),f'{name}_model age DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(metric_transformed.mean_difference(),f'{name}_model age Mean_difference value after')
    print_fairness_metrics(metric_transformed.disparate_impact(),f'{name}_model age DI value after')

    # otteniamo i nuovi pesi forniti dall'oggetto che mitigano i problemi di fairness
    post_pred = dataset_transformed.convert_to_dataframe()[0]    

    return post_pred

def test_eqodds(dataset,pred,name,first_message=False):
    # Attributi sensibili

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['sex'],
    )

    aif360_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['sex'],
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    metrics = ClassificationMetric(dataset=aif360_dataset,classified_dataset=aif360_pred,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model sex Eq. Odds difference')

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['age'],
    )

    aif360_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['num'],
        protected_attribute_names=['age'],
    )

    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    metrics = ClassificationMetric(dataset=aif360_dataset,classified_dataset=aif360_pred,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model age Eq. Odds difference')

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/aif360/heart_disease_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/heart_disease_postrocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
load_dataset()