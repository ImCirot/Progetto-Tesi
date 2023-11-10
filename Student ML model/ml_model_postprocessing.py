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
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
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
            sleep(1)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'Gender','Educational special needs',
        'Age at enrollment','Admission grade', 'International',
    ]

    feature_names = dataset.columns.tolist()

    feature_names.remove('Target')

    X = dataset[feature_names]
    y = dataset['Target']

    lr_model = pickle.load(open('./output_models/std_models/lr_student_model.sav','rb'))
    rf_model = pickle.load(open('./output_models/std_models/rf_student_model.sav','rb'))
    svm_model = pickle.load(open('./output_models/std_models/svm_student_model.sav','rb'))
    xgb_model = pickle.load(open('./output_models/std_models/xgb_student_model.sav','rb'))
    

    df_train, df_test, X_train,X_test,y_train,y_test = train_test_split(dataset,X,y,test_size=0.2,random_state=42)

    lr_pred = lr_model.predict(X_test)
    lr_df = X_test.copy(deep=True)
    lr_df['Target'] = lr_pred

    rf_pred = rf_model.predict(X_test)
    rf_df = X_test.copy(deep=True)
    rf_df['Target'] = rf_pred

    svm_pred = svm_model.predict(X_test)
    svm_df = X_test.copy(deep=True)
    svm_df['Target'] = svm_pred

    xgb_pred = xgb_model.predict(X_test)
    xgb_df = X_test.copy(deep=True)
    xgb_df['Target'] = xgb_pred

    print(f'######### Testing Fairness #########')
    lr_post_pred = test_fairness(df_test,lr_df)
    rf_post_pred = test_fairness(df_test,rf_df)
    svm_post_pred = test_fairness(df_test,svm_df)
    xgb_post_pred = test_fairness(df_test,xgb_df)
    
    print(f'######### Testing Risultati #########')
    validate(lr_model,lr_post_pred['Target'],'lr', X_test, y_test,True)
    validate(rf_model,rf_post_pred['Target'],'rf',X_test,y_test)
    validate(svm_model,svm_post_pred['Target'],'svm',X_test,y_test)
    validate(xgb_model,xgb_post_pred['Target'],'xgb',X_test,y_test)

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(model,fair_pred,model_type,X,y,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    
    accuracy = accuracy_score(y_pred=fair_pred,y_true=y)

    y_proba = model.predict_proba(X)[::,1]

    auc_score = roc_auc_score(y,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
        
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/postprocessing_models/student_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}\n")
        f.write(f'ROC-AUC Score: {round(auc_score,3)}\n')
        f.write('\n')

def test_fairness(dataset,pred):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Gender']
    )

    aif_gender_pred= BinaryLabelDataset(
        df=pred,
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
    
    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=gender_privileged_group, unprivileged_groups=gender_unprivileged_group,seed=42)

    eqoddspost.fit(aif_gender_dataset,aif_gender_pred)
    gender_trans_dataset = eqoddspost.predict(aif_gender_pred,threshold=0.8)

    gender_metric_trans = BinaryLabelDatasetMetric(dataset=gender_trans_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_trans.mean_difference(),'Gender mean_difference after')
    print_fairness_metrics(gender_metric_trans.disparate_impact(),"Gender DI after")

    new_dataset = gender_trans_dataset.convert_to_dataframe()[0]

    # verifichiamo se gli studenti normodotati ricevono predizioni positive maggiori rispetto
    # agli studenti con disabilità

    sn_aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
    )

    sn_aif_pred = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
    )

    sn_privileged_group = [{'Educational special needs': 0}]
    sn_unprivileged_group = [{'Educational special needs': 1}]

    sn_metric_og = BinaryLabelDatasetMetric(dataset=sn_aif_dataset,unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    print_fairness_metrics(sn_metric_og.mean_difference(),'Special Needs mean_difference before')
    print_fairness_metrics(sn_metric_og.disparate_impact(),"Special Needs DI before")

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=sn_privileged_group, unprivileged_groups=sn_unprivileged_group,seed=42)

    sn_trans_dataset = eqoddspost.fit_predict(sn_aif_dataset,sn_aif_pred,threshold=0.8)

    sn_metric_trans = BinaryLabelDatasetMetric(dataset=sn_trans_dataset,unprivileged_groups=sn_privileged_group, privileged_groups=sn_unprivileged_group)

    print_fairness_metrics(sn_metric_trans.mean_difference(),'Special Needs mean_difference after')
    print_fairness_metrics(sn_metric_trans.disparate_impact(),"Special Needs DI after")

    # salviamo il nuovo dataset ottenuto e i pesi rivalutati
    new_dataset = sn_trans_dataset.convert_to_dataframe()[0]

    # valutiamo ora eventuale disparità nelle età degli studenti
    # cerchiamo di valutare se uno studente con meno di 30 anni sia favorito rispetto a studenti in età più avanzata
    std_age_aif_dataset = StandardDataset(
        df=dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    std_age_aif_pred = StandardDataset(
        df=new_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    age_aif_dataset = BinaryLabelDataset(
        df=std_age_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
    )

    age_aif_pred = BinaryLabelDataset(
        df=std_age_aif_pred.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
    )

    age_privileged_group = [{'Age at enrollment': 1}]
    age_unprivileged_group = [{'Age at enrollment': 0}]

    age_metric_og = BinaryLabelDatasetMetric(dataset=age_aif_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_og.mean_difference(),'Age mean_difference before')
    print_fairness_metrics(age_metric_og.disparate_impact(),"Age DI before")

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=age_privileged_group, unprivileged_groups=age_unprivileged_group,seed=42)

    age_trans_dataset = eqoddspost.fit_predict(age_aif_dataset,age_aif_pred,threshold=0.8)

    age_metric_trans = BinaryLabelDatasetMetric(dataset=age_trans_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_trans.mean_difference(),'Age mean_difference after')
    print_fairness_metrics(age_metric_trans.disparate_impact(),"Age DI after")

    new_dataset = age_aif_dataset.convert_to_dataframe()[0]
    new_dataset_pred = age_trans_dataset.convert_to_dataframe()[0]

    # cerchiamo ora di stabilire se gli studenti internazionali sono svantaggiati
    int_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
    )
    
    int_aif_pred = BinaryLabelDataset(
        df=new_dataset_pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
    )

    int_privileged_group = [{'International': 0}]
    int_unprivileged_group = [{'International': 1}]

    int_metric_og = BinaryLabelDatasetMetric(dataset=int_aif_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_og.mean_difference(),'International mean_difference before')
    print_fairness_metrics(int_metric_og.disparate_impact(),"International DI before")

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=int_privileged_group, unprivileged_groups=int_unprivileged_group,seed=42)

    int_trans_dataset = eqoddspost.fit_predict(int_aif_dataset,int_aif_pred,threshold=0.8)

    int_metric_trans = BinaryLabelDatasetMetric(dataset=int_trans_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_trans.mean_difference(),'International mean_difference after')
    print_fairness_metrics(int_metric_trans.disparate_impact(),"International DI after")

    new_dataset = int_aif_dataset.convert_to_dataframe()[0]
    new_dataset_pred = int_trans_dataset.convert_to_dataframe()[0]

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

    std_grade_pred = StandardDataset(
        df=new_dataset_pred,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= mean_grade]
    )

    # costruiamo dataset binario per testare fairness della nostra scelta
    grade_aif_dataset = BinaryLabelDataset(
        df=std_grade_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
    )

    grade_aif_pred = BinaryLabelDataset(
        df=std_grade_pred.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
    )

    grade_privileged_group = [{'Admission grade': 1}]
    grade_unprivileged_group = [{'Admission grade': 0}]

    grade_metric_og = BinaryLabelDatasetMetric(dataset=grade_aif_dataset,unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_og.mean_difference(),'Grade mean_difference before')
    print_fairness_metrics(grade_metric_og.disparate_impact(),"Grade DI before")

    eqoddspost = CalibratedEqOddsPostprocessing(cost_constraint='weighted',privileged_groups=grade_privileged_group, unprivileged_groups=grade_unprivileged_group,seed=42)

    grade_trans_dataset = eqoddspost.fit_predict(grade_aif_dataset,grade_aif_pred,threshold=0.8)

    grade_metric_trans = BinaryLabelDatasetMetric(dataset=grade_trans_dataset, unprivileged_groups=grade_unprivileged_group, privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_trans.mean_difference(),'Grade mean_difference after')
    print_fairness_metrics(grade_metric_trans.disparate_impact(),"Grade DI after")

    postop_pred = grade_trans_dataset.convert_to_dataframe()[0]
    return postop_pred

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/student_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/student_postprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()