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
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime

@track_emissions(offline=True, country_iso_code="ITA")
def load_dataset():

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

    fair_dataset = df.copy(deep=True)

    sample_weights = test_fairness(df)

    fair_dataset['weights'] = sample_weights

    features = df.columns.tolist()
    features.remove('Target')

    target = ['Target']

    X = df[features]
    X_fair = fair_dataset[features]

    y = df[target]
    y_fair = df[target]

    sample_weights = fair_dataset['weights']

    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',LogisticRegression(class_weight={1:1,0:5}))
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',RandomForestClassifier(class_weight={1:1,0:5}))
    ])

    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',SVC(probability=True,class_weight={1:1,0:5}))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])
    
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

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_aif360_credit_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_aif360_credit_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_aif360_credit_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_aif360_credit_model.sav','wb'))
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
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/fair_models/aif360/credit_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f'ROC-AUC Score: {auc_score}\n')
        f.write('\n')

def test_fairness(dataset):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    # Attributi sensibili
    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_attribute_names,
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex_A93': 1}]
    unprivileged_groups = [{'sex_A93': 0}]

    # Calcolo della metrica sul dataset originale
    metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    
    # Se la metrica originale ottiene già valore 0.0, allora il dataset è gia equo e non ha bisogno di ulteriori operazioni
    if(metric_original.mean_difference() != 0.0):
        # Utilizzamo un operatore di bilanciamento offerto dall'API AIF360
        RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

        # Bilanciamo il dataset
        dataset_transformed = RW.fit_transform(aif360_dataset)
        # Ricalcoliamo la metrica
        metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(metric_original.mean_difference(),'Mean_difference value before', first_message=True)
    print_fairness_metrics(metric_original.disparate_impact(),'DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(metric_transformed.mean_difference(),'Mean_difference value after')
    print_fairness_metrics(metric_transformed.disparate_impact(),'DI value after')
    
    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(metric_original.num_positives(privileged=True),'Num. of positive instances of priv_group before')
    print_fairness_metrics(metric_original.num_positives(privileged=False),'Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(metric_transformed.num_positives(privileged=True),'Num. of positive instances of priv_group after')
    print_fairness_metrics(metric_transformed.num_positives(privileged=False),'Num. of positive instances of unpriv_group after')

    # otteniamo i nuovi pesi forniti dall'oggetto che mitigano i problemi di fairness
    sample_weights = dataset_transformed.instance_weights

    return sample_weights
    
def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/preprocessing/aif360/credit_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/aif360/credit_preprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)