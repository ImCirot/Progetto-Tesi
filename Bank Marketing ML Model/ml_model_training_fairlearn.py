from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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
from fairlearn.postprocessing import ThresholdOptimizer,plot_threshold_optimizer
import pickle
import xgboost as xgb

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    kf = KFold(n_splits=10)

    df_array = np.asarray(df)

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


    lr_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',LogisticRegression())
    ])

    rf_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',RandomForestClassifier())
    ])

    svm_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',SVC(probability=True))
    ])

    xgb_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])

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

    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    i = 0

    for train_index, test_index in kf.split(df_array):
        i = i + 1

        print(f'\n######### Inizio {i} iterazione #########\n')

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]
        g_train = g.iloc[train_index]

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]
        g_test = g.iloc[test_index]

        lr_model_pipeline.fit(X_train,y_train)
        rf_model_pipeline.fit(X_train,y_train)
        svm_model_pipeline.fit(X_train,y_train)
        xgb_model_pipeline.fit(X_train,y_train)

        validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
        validate(rf_model_pipeline,'rf','std',i,X_test,y_test)
        validate(svm_model_pipeline,'svm','std',i,X_test,y_test)
        validate(xgb_model_pipeline,'xgb','std',i,X_test,y_test)

        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train)

        validate(lr_fair_model_pipeline,'lr','fair',i,X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,'rf','fair',i,X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,'svm','fair',i,X_fair_test,y_fair_test)
        validate(xgb_fair_model_pipeline,'xgb','fair',i,X_fair_test,y_fair_test)

        lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
        rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
        svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
        xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

        validate_postop(lr_threshold,'lr',i,X_test,y_test,g_test)
        validate_postop(rf_threshold,'rf',i,X_test,y_test,g_test)
        validate_postop(svm_threshold,'svm',i,X_test,y_test,g_test)
        validate_postop(xgb_threshold,'xgb',i,X_test,y_test,g_test)

        print(f'\n######### Fine {i} iterazione #########\n')

    lr_std_pred = lr_model_pipeline.predict(X)
    lr_fair_pred = lr_fair_model_pipeline.predict(X)
    lr_threshold_pred = lr_threshold.predict(X,sensitive_features=g)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_fair_pred = rf_fair_model_pipeline.predict(X)
    rf_threshold_pred = rf_threshold.predict(X,sensitive_features=g)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_fair_pred = svm_fair_model_pipeline.predict(X)
    svm_threshold_pred = svm_threshold.predict(X,sensitive_features=g)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X)
    xgb_threshold_pred = xgb_threshold.predict(X,sensitive_features=g)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_fair': lr_fair_pred,
        'lr_threshold': lr_threshold_pred,
        'rf_std': rf_std_pred,
        'rf_fair':rf_fair_pred,
        'rf_threshold':rf_threshold_pred,
        'svm_std': svm_std_pred,
        'svm_fair': svm_fair_pred,
        'svm_threshold': svm_threshold_pred,
        'xgb_std': xgb_std_pred,
        'xgb_fair': xgb_fair_pred,
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

        with open('./reports/fairness_reports/postprocessing/fairlearn/bank_model_DI.txt',open_type) as f:
            f.write(f'{name}_marital DI: {marital_DI}\n')
            f.write(f'{name}_education DI: {education_DI}\n')

    print(f'######### Inizio stesura report finale #########')
    with open('./reports/final_scores/fairlearn/bank_scores.txt','w') as f:
        f.write(f'LR std model: {str(lr_model_pipeline.score(X,y))}\n')
        f.write(f'RF std model: {str(rf_model_pipeline.score(X,y))}\n')
        f.write(f'SVM std model: {str(svm_model_pipeline.score(X,y))}\n')
        f.write(f'XGB std model: {str(xgb_model_pipeline.score(X,y))}\n')
        
        f.write(f'LR fair model: {str(lr_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'RF fair model: {str(rf_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'SVM fair model: {str(svm_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'XGB fair model: {str(xgb_fair_model_pipeline.score(X_fair,y_fair))}\n')
    
    print(f'######### Inizio salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_fairlearn_bank_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_fairlearn_bank_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_fairlearn_bank_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_fairlearn_bank_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_fairlearn_bank_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_fairlearn_abank_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_fairlearn_bank_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_fairlearn_bank_model.sav','wb'))
    pickle.dump(lr_threshold,open('./output_models/postop_models/threshold_lr_fairlearn_bank_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/postop_models/threshold_rf_fairlearn_bank_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/postop_models/threshold_svm_fairlearn_bank_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/postop_models/threshold_xgb_fairlearn_bank_model.sav','wb'))
    
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

    show_correlation_heatmap(dataset=dataset,title='original dataset')
    show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset

def show_correlation_heatmap(dataset,title):

    plt.figure(figsize=(10,8))
    sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    plt.title(title)
    plt.show()

def validate(ml_model,ml_type,ml_vers,index,X_test,y_test):
    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test,pred)

    report = classification_report(y_test,pred)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"

    with  open(f"./reports/{ml_vers}_models/fairlearn/bank/{ml_type}_bank_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')

    with  open(f"./reports/{ml_vers}_models/fairlearn/bank/{ml_type}_bank_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def validate_postop(ml_model,model_type,index,X_test,y_test,g_test):
    pred = ml_model.predict(X_test,sensitive_features=g_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/postop_models/fairlearn/bank/{model_type}_bank_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/postop_models/fairlearn/bank/{model_type}_bank_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

load_dataset()