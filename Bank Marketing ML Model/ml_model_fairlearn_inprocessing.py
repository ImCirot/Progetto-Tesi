from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from fairlearn.metrics import *
from fairlearn.postprocessing import ThresholdOptimizer,plot_threshold_optimizer
import pickle
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    training_and_testing_models(df)

def training_and_testing_models(df):
    feature_names = df.columns.tolist()

    feature_names.remove('y')

    protected_features_names = [
         'marital_divorced','marital_married','marital_single',
         'education_primary','education_secondary','education_tertiary'
    ] 

    marital_feature_names = [
        'marital_divorced','marital_married','marital_single'
    ]

    education_feature_names = [
        'education_primary','education_secondary','education_tertiary'
    ]

    X = df[feature_names]
    
    y = df['y']
    
    g = df[protected_features_names]
    g_marital = df[marital_feature_names]
    g_education = df[education_feature_names]

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_bank_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_bank_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_bank_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_bank_model.sav','rb'))


    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False
    )

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
    rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
    svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
    xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

    print(f'######### Testing modelli #########')
    validate(lr_threshold,'lr',X_test,y_test,g_test,True)
    validate(rf_threshold,'rf',X_test,y_test,g_test)
    validate(svm_threshold,'svm',X_test,y_test,g_test)
    validate(xgb_threshold,'xgb',X_test,y_test,g_test)

    print(f'######### Testing Fairness #########')
    lr_std_pred = lr_model_pipeline.predict(X)
    lr_threshold_pred = lr_threshold.predict(X,sensitive_features=g)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_threshold_pred = rf_threshold.predict(X,sensitive_features=g)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_threshold_pred = svm_threshold.predict(X,sensitive_features=g)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_threshold_pred = xgb_threshold.predict(X,sensitive_features=g)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_threshold': lr_threshold_pred,
        'rf_std': rf_std_pred,
        'rf_threshold':rf_threshold_pred,
        'svm_std': svm_std_pred,
        'svm_threshold': svm_threshold_pred,
        'xgb_std': xgb_std_pred,
        'xgb_threshold': xgb_threshold_pred
    }

    start = True

    for name,prediction in predictions.items():

        marital_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g_marital)
        education_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g_education)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/inprocessing/fairlearn/bank_report.txt',open_type) as f:
            f.write(f'{name}_marital DI: {round(marital_DI,3)}\n')
            f.write(f'{name}_education DI: {round(education_DI,3)}\n')

    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_threshold,open('./output_models/inprocess_models/threshold_lr_fairlearn_bank_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/inprocess_models/threshold_rf_fairlearn_bank_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/inprocess_models/threshold_svm_fairlearn_bank_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/inprocess_models/threshold_xgb_fairlearn_bank_model.sav','wb'))
    
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')


def show_correlation_heatmap(dataset,title):

    plt.figure(figsize=(10,8))
    sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    plt.title(title)
    plt.show()

def validate(ml_model,model_type,X_test,y_test,g_test,first=False):
    pred = ml_model.predict(X_test,sensitive_features=g_test)

    accuracy = accuracy_score(y_test, pred)

    y_proba = ml_model.estimator.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/inprocessing_models/fairlearn/bank_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/fairlearn/bank_inprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')


start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)