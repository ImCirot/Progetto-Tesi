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
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
import pickle
from datetime import datetime
from time import sleep


def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Student Dataset/dataset.csv')

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
            sleep(120)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'Gender','Educational special needs',
        'Age at enrollment','Admission grade', 'International',
    ]

    sample_weights = test_fairness(dataset)

    fair_dataset = dataset.copy(deep=True)

    fair_dataset['weights'] = sample_weights

    feature_names = dataset.columns.tolist()

    feature_names.remove('Target')

    X = dataset[feature_names]
    y = dataset['Target']

    X_fair = fair_dataset[feature_names]
    y_fair = fair_dataset['Target']
    weights = fair_dataset['weights']

    # settiamo i nostri modelli sul dataset fair
    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',LogisticRegression())
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',RandomForestClassifier())
    ])

    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',SVC(probability=True))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic',random_state=42))
    ])

    X_fair_train, X_fair_test, y_fair_train, y_fair_test, weights_train, weights_test = train_test_split(X_fair,y_fair,weights,test_size=0.2,random_state=42)

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

    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_aif360_student_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_aif360_student_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_aif360_student_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_aif360_student_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    pred = ml_model.predict(X_test)
    
    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
        
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/preprocessing_models/aif360/student_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')

def test_fairness(dataset):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Gender']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    gender_privileged_group = [{'Gender': 1}]
    gender_unprivileged_group = [{'Gender': 0}]

    gender_metric_og = BinaryLabelDatasetMetric(dataset=aif_gender_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_og.mean_difference(),'Gender mean_difference before',first_message=True)
    print_fairness_metrics(gender_metric_og.disparate_impact(),"Gender DI before")
    
    gender_RW = Reweighing(unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    gender_trans_dataset = gender_RW.fit_transform(aif_gender_dataset)

    gender_metric_trans = BinaryLabelDatasetMetric(dataset=gender_trans_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_trans.mean_difference(),'Gender mean_difference after')
    print_fairness_metrics(gender_metric_trans.disparate_impact(),"Gender DI after")

    new_dataset = gender_trans_dataset.convert_to_dataframe()[0]
    sample_weights = gender_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # verifichiamo se gli studenti normodotati ricevono predizioni positive maggiori rispetto
    # agli studenti con disabilità
    sn_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
        instance_weights_name=['weights']
    )

    sn_privileged_group = [{'Educational special needs': 0}]
    sn_unprivileged_group = [{'Educational special needs': 1}]

    sn_metric_og = BinaryLabelDatasetMetric(dataset=sn_aif_dataset,unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    print_fairness_metrics(sn_metric_og.mean_difference(),'Special Needs mean_difference before')
    print_fairness_metrics(sn_metric_og.disparate_impact(),"Special Needs DI before")

    sn_RW = Reweighing(unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    sn_trans_dataset = sn_RW.fit_transform(sn_aif_dataset)

    sn_metric_trans = BinaryLabelDatasetMetric(dataset=sn_trans_dataset,unprivileged_groups=sn_unprivileged_group, privileged_groups=sn_privileged_group)

    print_fairness_metrics(sn_metric_trans.mean_difference(),'Special Needs mean_difference after')
    print_fairness_metrics(sn_metric_trans.disparate_impact(),"Special Needs DI after")

    # salviamo il nuovo dataset ottenuto e i pesi rivalutati
    new_dataset = sn_trans_dataset.convert_to_dataframe()[0]

    sample_weights = sn_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # valutiamo ora eventuale disparità nelle età degli studenti
    # cerchiamo di valutare se uno studente con meno di 30 anni sia favorito rispetto a studenti in età più avanzata
    std_age_aif_dataset = StandardDataset(
        df=new_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    mod_age_dataset = std_age_aif_dataset.convert_to_dataframe()[0]

    mod_age_dataset['weights'] = sample_weights

    age_aif_dataset = BinaryLabelDataset(
        df=std_age_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
        instance_weights_name=['weights'],
    )

    age_privileged_group = [{'Age at enrollment': 1}]
    age_unprivileged_group = [{'Age at enrollment': 0}]

    age_metric_og = BinaryLabelDatasetMetric(dataset=age_aif_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_og.mean_difference(),'Age mean_difference before')
    print_fairness_metrics(age_metric_og.disparate_impact(),"Age DI before")

    age_RW = Reweighing(unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    age_trans_dataset = age_RW.fit_transform(age_aif_dataset)

    age_metric_trans = BinaryLabelDatasetMetric(dataset=age_trans_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_trans.mean_difference(),'Age mean_difference after')
    print_fairness_metrics(age_metric_trans.disparate_impact(),"Age DI after")

    new_dataset = age_trans_dataset.convert_to_dataframe()[0]

    sample_weights = age_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # cerchiamo ora di stabilire se gli studenti internazionali sono svantaggiati
    int_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
        instance_weights_name=['weights'],
    )

    int_privileged_group = [{'International': 0}]
    int_unprivileged_group = [{'International': 1}]

    int_metric_og = BinaryLabelDatasetMetric(dataset=int_aif_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_og.mean_difference(),'International mean_difference before')
    print_fairness_metrics(int_metric_og.disparate_impact(),"International DI before")

    int_RW = Reweighing(unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    int_trans_dataset = int_RW.fit_transform(int_aif_dataset)

    int_metric_trans = BinaryLabelDatasetMetric(dataset=int_trans_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_trans.mean_difference(),'International mean_difference after')
    print_fairness_metrics(int_metric_trans.disparate_impact(),"International DI after")

    new_dataset = int_trans_dataset.convert_to_dataframe()[0]

    sample_weights = int_trans_dataset.instance_weights
    
    new_dataset['weights'] = sample_weights

    # con questo comando stampiamo il voto più alto e più basso ottenuti all'interno del campione utilizzato.
    # Otteniamo anche la media, utilizzeremo la media aritmentica dei voti come valore per dividere il dataset in studenti con voto sotto la media
    # e sopra la media.
    # print(new_dataset['Admission grade'].mean(), new_dataset['Admission grade'].max(), new_dataset['Admission grade'].min())
    mean_grade = new_dataset['Admission grade'].mean()
    
    # infine, cerchiamo di valutare se gli studenti con un basso voto di ammissione siano sfavoriti per predizione positiva
    # selezioniamo appunto come favoriti tutti gli studenti il cui voto di ammissione supera il voto medio di ammissione
    std_grade_dataset = StandardDataset(
        df=new_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= mean_grade]
    )

    new_dataset = std_grade_dataset.convert_to_dataframe()[0]

    new_dataset['weights'] = sample_weights

    # costruiamo dataset binario per testare fairness della nostra scelta
    grade_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
        instance_weights_name=['weights'],
    )

    grade_privileged_group = [{'Admission grade': 1}]
    grade_unprivileged_group = [{'Admission grade': 0}]

    grade_metric_og = BinaryLabelDatasetMetric(dataset=grade_aif_dataset,unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_og.mean_difference(),'Grade mean_difference before')
    print_fairness_metrics(grade_metric_og.disparate_impact(),"Grade DI before")

    grade_RW = Reweighing(unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    grade_trans_dataset = grade_RW.fit_transform(grade_aif_dataset)

    grade_metric_trans = BinaryLabelDatasetMetric(dataset=grade_trans_dataset, unprivileged_groups=grade_unprivileged_group, privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_trans.mean_difference(),'Grade mean_difference after')
    print_fairness_metrics(grade_metric_trans.disparate_impact(),"Grade DI after")

    # restituiamo infine i pesi ottenuti da quest'ultima iterazione
    sample_weights = grade_trans_dataset.instance_weights

    return sample_weights


def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/preprocessing/aif360/student_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/student_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()