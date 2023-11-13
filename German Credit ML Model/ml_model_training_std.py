import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset,StandardDataset
from aif360.metrics import ClassificationMetric
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep


def load_dataset():
    ## funzione di load del dataset
    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    df['Target'] = df['Target'].replace(2,0)

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
    features.remove('Target')

    target = ['Target']

    X = df[features]
    y = df[target]

    # Creiamo due pipeline che effettuano delle ulteriori operazioni di scaling dei dati per addestriare il modello
    # in particolare la pipeline standard sarà addestrata sui dati as-is
    # mentre la fair pipeline verrà addestrata su dati sui vengono applicate strategie di fairness
    # volte a rimuovere discriminazione e bias nel dataset di training
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    # Strategia KFold
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    print(f'######### Training modelli #########')
    # fit del modello sul training set dell'i-esima iterazione
    lr_model_pipeline.fit(X_train,y_train.values.ravel())
    rf_model_pipeline.fit(X_train,y_train.values.ravel())
    svm_model_pipeline.fit(X_train,y_train.values.ravel())
    xgb_model_pipeline.fit(X_train,y_train.values.ravel())

    # Stampiamo metriche di valutazione per il modello
    print(f'######### Testing modelli #########')
    validate(lr_model_pipeline,'lr', X_test, y_test,True)
    validate(rf_model_pipeline,'rf',X_test,y_test)
    validate(svm_model_pipeline,'svm',X_test,y_test)
    validate(xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Testing fairness #########')
    X_test_df = X_test.copy(deep=True)
    X_test_df['Target'] = y_test['Target']

    lr_pred = X_test.copy(deep=True)
    lr_pred['Target'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test.copy(deep=True)
    rf_pred['Target'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test.copy(deep=True)
    svm_pred['Target'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test.copy(deep=True)
    xgb_pred['Target'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,first_message=True)
    eq_odds_fair_report(X_test_df,rf_pred)
    eq_odds_fair_report(X_test_df,svm_pred)
    eq_odds_fair_report(X_test_df,xgb_pred)
    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_credit_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_credit_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_credit_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')
            
def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input
    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/std_models/credit_metrics_report.txt",open_type) as f:
        f.write(f'{model_type}\n')
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def eq_odds_fair_report(dataset,prediction,first_message=False):
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

    aif360_pred = BinaryLabelDataset(
        df=prediction,
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

    metrics = ClassificationMetric(dataset=aif360_dataset,classified_dataset=aif360_pred,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_eqodds_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),'Eq. Odds difference from std classifier',first_message)


def print_eqodds_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/eqodds/std/aif360/credit_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')


def print_time(time,index):

    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    
    with open('./reports/time_reports/std/credit_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

# Chiamata funzione iniziale di load dataset
load_dataset()