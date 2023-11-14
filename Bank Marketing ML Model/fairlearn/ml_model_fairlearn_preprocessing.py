from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import *
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_and_testing_models(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(300)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
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

    fair_dataset = fairness_preprocess_op(dataset=df, protected_features_names=protected_features_names)

    X = df[feature_names]
    X_fair = fair_dataset[feature_names]
    
    y = df['y']
    y_fair = fair_dataset['y']
    
    g = df[protected_features_names]
    g_marital = df[marital_feature_names]
    g_education = df[education_feature_names]

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_bank_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_bank_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_bank_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_bank_model.sav','rb'))

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
        ('model',SVC(probability=True))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)
    X_fair_train, X_fair_test, y_fair_train, y_fair_test = train_test_split(X_fair,y_fair,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train)

    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)
    
    print(f'######### Testing Fairness #########')
    lr_std_pred = lr_model_pipeline.predict(X)
    lr_fair_pred = lr_fair_model_pipeline.predict(X)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_fair_pred = rf_fair_model_pipeline.predict(X)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_fair_pred = svm_fair_model_pipeline.predict(X)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_fair': lr_fair_pred,
        'rf_std': rf_std_pred,
        'rf_fair':rf_fair_pred,
        'svm_std': svm_std_pred,
        'svm_fair': svm_fair_pred,
        'xgb_std': xgb_std_pred,
        'xgb_fair': xgb_fair_pred,
    }

    start = True

    for name,prediction in predictions.items():

        DI_value = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g)
        mean_diff = demographic_parity_difference(y_true=y,y_pred=prediction,sensitive_features=g)
        eq_odds_diff = equalized_odds_difference(y_true=y,y_pred=prediction,sensitive_features=g)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/preprocessing/fairlearn/bank_report.txt',open_type) as f:
            f.write(f'{name} DI: {round(DI_value,3)}\n')
            f.write(f'{name} mean diff: {round(mean_diff,3)}\n')
            f.write(f'{name} eq. odds diff: {round(eq_odds_diff,3)}\n')
    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_fairlearn_bank_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_fairlearn_bank_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_fairlearn_bank_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_fairlearn_bank_model.sav','wb'))
    
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def fairness_preprocess_op(dataset,protected_features_names):
    features_names = dataset.columns.tolist()

    for feature in protected_features_names:
        features_names.remove(feature)

    corr_remover = CorrelationRemover(sensitive_feature_ids=protected_features_names,alpha=1.0)

    fair_dataset = corr_remover.fit_transform(dataset)

    fair_dataset = pd.DataFrame(
        fair_dataset,columns=features_names
    )

    fair_dataset[protected_features_names] = dataset[protected_features_names]
    fair_dataset['y'] = dataset['y']
    fair_dataset = fair_dataset[dataset.columns.tolist()]

    # show_correlation_heatmap(dataset=dataset,title='original dataset')
    # show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset

def show_correlation_heatmap(dataset,title):

    plt.figure(figsize=(10,8))
    sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    plt.title(title)
    plt.show()

def validate(ml_model,ml_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"

    with  open(f"./reports/preprocessing_models/fairlearn/bank_metrics_report.txt",open_type) as f:
        f.write(f"{ml_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    with open('./reports/time_reports/fairlearn/bank_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()