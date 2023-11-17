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
from sklearn.svm import LinearSVC
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
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')
    
@track_emissions(offline=True, country_iso_code="ITA")
def training_and_testing_model(df):
    ## Funzione per il training e testing del modello scelto

    fair_dataset = df.copy(deep=True)

    sample_weights = test_fairness(df)

    fair_dataset['weights'] = sample_weights

    features = df.columns.tolist()
    features.remove('num')

    target = ['num']

    X = df[features]
    X_fair = fair_dataset[features]

    y = df[target]
    y_fair = df[target]

    sample_weights = fair_dataset['weights']

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_heart_disease_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_heart_disease_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_heart_disease_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_heart_disease_model.sav','rb'))

    protected_attribute_names = [
       'sex','age'
    ]

    age_feature = ['age']
    sex_feature = ['sex']

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
        ('model',LinearSVC(dual='auto'))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_fair_train, X_fair_test, y_fair_train, y_fair_test, sample_weights_train, sample_weights_test = train_test_split(X_fair,y_fair,sample_weights,test_size=0.2,random_state=42)
    
    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)

    # Stampiamo metriche di valutazione per il modello
    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr', X_fair_test, y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    print('######### Testing Fairness #########')
    X_test_fair_df = X_fair_test.copy(deep=True)
    X_test_fair_df['num'] = y_fair_test['num']

    X_test_df = X_test.copy(deep=True)
    X_test_df['num'] = y_test['num']

    lr_fair_pred = X_test_df.copy(deep=True)
    lr_fair_pred['num'] = lr_fair_model_pipeline.predict(X_test)

    rf_fair_pred =  X_test_df.copy(deep=True)
    rf_fair_pred['num'] = rf_fair_model_pipeline.predict(X_test)

    svm_fair_pred =  X_test_df.copy(deep=True)
    svm_fair_pred['num'] = svm_fair_model_pipeline.predict(X_test)

    xgb_fair_pred =  X_test_df.copy(deep=True)
    xgb_fair_pred['num'] = xgb_fair_model_pipeline.predict(X_test)

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['num'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test_df.copy(deep=True)
    rf_pred['num'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test_df.copy(deep=True)
    svm_pred['num'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test_df.copy(deep=True)
    xgb_pred['num'] = xgb_model_pipeline.predict(X_test)

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
        eq_odds_fair_report(X_test_fair_df,prediction,name)


    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/aif360/lr_heart_disease_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/aif360/rf_heart_disease_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/aif360/svm_heart_disease_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/aif360/xgb_heart_disease_model.sav','wb'))
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
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/preprocessing_models/aif360/heart_disease_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'F1 score: {round(f1,3)}\n')
        f.write(f"Precision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')

def test_fairness(dataset):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

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

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex': 1} | {'age': 1}]
    unprivileged_groups = [{'sex': 0, 'age': 0}]

    sex_privileged_groups = [{'sex':1}]
    sex_unprivileged_groups = [{'sex':0}]

    age_privileged_groups = [{'age':1}]
    age_unprivileged_groups = [{'age':0}]

    # Calcolo della metrica sul dataset originale
    sex_metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups) 

    age_metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=age_unprivileged_groups, privileged_groups=age_privileged_groups)

    # Utilizzamo un operatore di bilanciamento offerto dall'API AIF360
    RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

    # Bilanciamo il dataset
    dataset_transformed = RW.fit_transform(aif360_dataset)
    # Ricalcoliamo la metrica
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups)

    age_metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=age_unprivileged_groups, privileged_groups=age_privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(sex_metric_original.mean_difference(),'sex Mean_difference value before', first_message=True)
    print_fairness_metrics(sex_metric_original.disparate_impact(),'sex DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(sex_metric_transformed.mean_difference(),'sex Mean_difference value after')
    print_fairness_metrics(sex_metric_transformed.disparate_impact(),'sex DI value after')

    print_fairness_metrics(age_metric_original.mean_difference(),'age Mean_difference value before')
    print_fairness_metrics(age_metric_original.disparate_impact(),'age DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(age_metric_transformed.mean_difference(),'age Mean_difference value after')
    print_fairness_metrics(age_metric_transformed.disparate_impact(),'age DI value after')

    # otteniamo i nuovi pesi forniti dall'oggetto che mitigano i problemi di fairness
    sample_weights = dataset_transformed.instance_weights

    new_dataset = dataset_transformed.convert_to_dataframe()[0]

    return sample_weights

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

    print_fairness_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model sex Eq. Odds difference')

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
    with open(f"./reports/fairness_reports/preprocessing/aif360/heart_disease_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/heart_disease_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
load_dataset()