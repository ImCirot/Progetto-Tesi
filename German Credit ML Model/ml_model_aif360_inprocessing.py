import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime

@track_emissions(offline=True, country_iso_code="ITA")
def load_dataset():
    ##funzione di load del dataset

    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    df['Target'] = df['Target'].replace(2,0)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        training_and_testing_model(df)

def training_and_testing_model(df):
    ## Funzione per il training e testing del modello scelto
    features = df.columns.tolist()
    features.remove('Target')

    target = ['Target']

    X = df[features]

    y = df[target]

    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    post_lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_train = processing_fairness(df,X_train,y_train,protected_attribute_names)

    X_postop_train = processed_train[features]
    y_postop_train = processed_train['Target']

    print(f'######### Training modelli #########')
    post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
    post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
    post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
    post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

    print(f'######### Testing modelli #########')
    validate_postop(post_lr_model_pipeline,'lr',X_test,y_test,True)
    validate_postop(post_rf_model_pipeline,'rf',X_test,y_test)
    validate_postop(post_svm_model_pipeline,'svm',X_test,y_test)
    validate_postop(post_xgb_model_pipeline,'xgb',X_test,y_test)
    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocess_models/lr_aif360_credit_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocess_models/rf_aif360_credit_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocess_models/svm_aif360_credit_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocess_models/xgb_aif360_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')
            
def processing_fairness(dataset,X_set,y_set,protected_features):

    fair_classifier = MetaFairClassifier(type='sr')

    train_dataset = pd.DataFrame(X_set)

    train_dataset['Target'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'sex_A93': 1}]
    unprivileged_groups = [{'sex_A93': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)
    
    print_inproc_metrics(metrics_og.mean_difference(),f'Gender Mean difference pre inprocessing',first_message=True)
    print_inproc_metrics(metrics_og.disparate_impact(),f'Gender DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_inproc_metrics(metrics_trans.mean_difference(),f'Gender Mean difference post inprocessing')
    print_inproc_metrics(metrics_trans.disparate_impact(),f'Gender DI post inprocessing')

    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train

def print_inproc_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/aif360/credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def validate_postop(ml_model,model_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f'./reports/inprocessing_models/aif360/credit_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')
            
def print_time(time):
    with open('./reports/time_reports/aif360/credit_inprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)