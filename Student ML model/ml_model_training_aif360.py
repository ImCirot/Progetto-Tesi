from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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

def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Student Dataset/dataset.csv')

    # drop ID dal dataframe
    df.drop('ID', inplace=True,axis=1)

    # richiamo funzione di training e testing dei modelli
    training_testing_models(df)

def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'Gender','Educational special needs',
        'Age at enrollment','Admission grade', 'International',
    ]

    # (fair_dataset,sample_weights) = 
    test_fairness(dataset)


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


def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/student_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

load_dataset()