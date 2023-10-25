from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import MetaFairClassifier
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    df_fair = df.copy(deep=True)

    instance_weights = test_fairness(df)
    
    df_fair['weights'] = instance_weights

    kf = KFold(n_splits=10)

    df_array = np.asarray(df)

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    protected_features = [
        'marital_divorced','marital_married','marital_single','education_primary','education_secondary','education_tertiary'
    ]

    X = df[feature_names]
    y = df['y']

    X_fair = df_fair[feature_names]
    y_fair = df_fair['y']
    sample_weights = df_fair['weights']

    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

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
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])

    i = 0

    for train_index, test_index in kf.split(df_array):
        i = i + 1

        print(f'\n######### Inizio {i} iterazione #########\n')
        
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        sample_weights_train = sample_weights.iloc[train_index]
        
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        lr_model_pipeline.fit(X_train,y_train)
        rf_model_pipeline.fit(X_train,y_train)
        svm_model_pipeline.fit(X_train,y_train)
        xgb_model_pipeline.fit(X_train,y_train)

        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
        xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)

        validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
        validate(rf_model_pipeline,'rf','std',i,X_test,y_test)
        validate(svm_model_pipeline,'svm','std',i,X_test,y_test)
        validate(xgb_model_pipeline,'xgb','std',i,X_test,y_test)

        validate(lr_fair_model_pipeline,'lr','fair',i,X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,'rf','fair',i,X_fair_test,y_fair_test)
        validate(svm_model_pipeline,'svm','fair',i,X_fair_test,y_fair_test)
        validate(xgb_fair_model_pipeline,'xgb','fair',i,X_fair_test,y_fair_test)

        processed_train = processing_fairness(df,X_train,y_train,protected_features,i)

        X_postop_train = processed_train[feature_names]
        y_postop_train = processed_train['y']

        post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
        post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
        post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
        post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

        validate_postop(post_lr_model_pipeline,'lr',i,X_test,y_test)
        validate_postop(post_rf_model_pipeline,'rf',i,X_test,y_test)
        validate_postop(post_svm_model_pipeline,'svm',i,X_test,y_test)
        validate_postop(post_xgb_model_pipeline,'xgb',i,X_test,y_test)

        print(f'\n######### Fine {i} iterazione #########\n')

    print(f'######### Inizio stesura report finale #########')
    with open('./reports/final_scores/aif360/bank_scores.txt','w') as f:
        f.write(f'LR std model: {str(lr_model_pipeline.score(X,y))}\n')
        f.write(f'RF std model: {str(rf_model_pipeline.score(X,y))}\n')
        f.write(f'SVM std model: {str(svm_model_pipeline.score(X,y))}\n')
        f.write(f'XGB std model: {str(xgb_model_pipeline.score(X,y))}\n')
        
        f.write(f'LR fair model: {str(lr_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'RF fair model: {str(rf_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'SVM fair model: {str(svm_fair_model_pipeline.score(X_fair,y_fair))}\n')
        f.write(f'XGB fair model: {str(xgb_fair_model_pipeline.score(X_fair,y_fair))}\n')

        f.write(f'LR post model: {str(post_lr_model_pipeline.score(X,y))}\n')
        f.write(f'RF post model: {str(post_rf_model_pipeline.score(X,y))}\n')
        f.write(f'SVM post model: {str(post_svm_model_pipeline.score(X,y))}\n')
        f.write(f'XGB post model: {str(post_xgb_model_pipeline.score(X,y))}\n')

    print(f'######### Inizio salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_aif360_bank_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_aif360_bank_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_aif360_bank_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_aif360_bank_model.sav','wb'))

    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_aif360_bank_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_aif360_bank_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_aif360_bank_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_aif360_bank_model.sav','wb'))

    pickle.dump(post_lr_model_pipeline,open('./output_models/postop_models/lr_aif360_bank_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/postop_models/rf_aif360_bank_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/postop_models/svm_aif360_bank_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/postop_models/xgb_aif360_bank_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def processing_fairness(dataset,X_set,y_set,protected_features,index):

    fair_classifier = MetaFairClassifier(type='sr')

    train_dataset = pd.DataFrame(X_set)

    train_dataset['y'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=protected_features
    )

    privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    unprivileged_groups = [{'marital_divorced': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    if index == 1:
        first_message = True
    else:
        first_message = False
    
    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Marital Mean difference pre inprocessing',first_message=first_message)
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Marital DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Marital Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Marital DI post inprocessing')

    privileged_groups = [{'education_primary': 1}]
    unprivileged_groups = [{'education_primary': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Education Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Education DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)


    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train

def validate_postop(ml_model,model_type,index,X_test,y_test):
    pred = ml_model.predict(X_test)

    report = classification_report(y_pred=pred,y_true=y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if index == 1:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/postop_models/aif360/bank/{model_type}_bank_metrics_report.txt',open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write('Metriche di valutazione:')
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def print_postop_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/aif360/bank_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

def test_fairness(dataset):
    maritial_features = [
        'marital_divorced','marital_married','marital_single'
    ]

    education_features = [
        'education_primary','education_secondary','education_tertiary'
    ]

    marital_df_aif360 = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=maritial_features
    )

    marital_privileged_groups = [{'marital_single': 1},{'marital_married': 1}]
    marital_unprivileged_groups = [{'marital_divorced': 1}]

    marital_metric_original = BinaryLabelDatasetMetric(dataset=marital_df_aif360,unprivileged_groups=marital_unprivileged_groups, privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_original.disparate_impact(),'(Marital) DI before', first_message=True)
    print_fairness_metrics(marital_metric_original.mean_difference(),'(Marital) mean_difference before')

    marital_RW = Reweighing(privileged_groups=marital_privileged_groups,unprivileged_groups=marital_unprivileged_groups)

    marital_df_trans = marital_RW.fit_transform(marital_df_aif360)

    marital_metric_trans =  BinaryLabelDatasetMetric(dataset=marital_df_trans,unprivileged_groups=marital_unprivileged_groups,privileged_groups=marital_privileged_groups)

    print_fairness_metrics(marital_metric_trans.disparate_impact(),'(Marital) DI after')
    print_fairness_metrics(marital_metric_trans.mean_difference(),'(Marital) mean_difference after')

    df_mod = marital_df_trans.convert_to_dataframe()[0]

    sample_weights = marital_df_trans.instance_weights

    df_mod['weights'] = sample_weights

    ed_df_aif360 = BinaryLabelDataset(
        df=df_mod,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features,
        instance_weights_name =['weights']
    )

    ed_privileged_groups = [{'education_primary': 1}]
    ed_unprivileged_groups = [{'education_primary': 0}]

    ed_metric_original = BinaryLabelDatasetMetric(dataset=ed_df_aif360,unprivileged_groups=ed_unprivileged_groups, privileged_groups=ed_privileged_groups)

    print_fairness_metrics(ed_metric_original.disparate_impact(),'(Ed.) DI before')
    print_fairness_metrics(ed_metric_original.mean_difference(),'(Ed.) Mean_difference before')

    ed_RW = Reweighing(privileged_groups=ed_privileged_groups,unprivileged_groups=ed_unprivileged_groups)

    ed_df_trans = ed_RW.fit_transform(dataset=ed_df_aif360)

    ed_metric_trans =  BinaryLabelDatasetMetric(dataset=ed_df_trans,unprivileged_groups=ed_unprivileged_groups,privileged_groups=ed_privileged_groups)

    print_fairness_metrics(ed_metric_trans.disparate_impact(),'(Ed.) DI after')
    print_fairness_metrics(ed_metric_trans.mean_difference(),'(Ed.) Mean_difference after')

    sample_weights = ed_df_trans.instance_weights

    return sample_weights


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

    with  open(f"./reports/{ml_vers}_models/aif360/bank/{ml_type}_marketing_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')

    with  open(f"./reports/{ml_vers}_models/aif360/bank/{ml_type}_marketing_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/marketing_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

load_dataset()