from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
import numpy as np
import pandas as pd

def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    (df_fair,instance_weights) = test_fairness(df)
    df_fair['weights'] = instance_weights

    kf = KFold(n_splits=10)

    df_array = np.asarray(df)

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    X = df[feature_names]
    y = df['y']

    X_fair = df_fair[feature_names]
    y_fair = df_fair['y']
    sample_weights = df_fair['weights']

    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))

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
    
    i = 0

    for train_index, test_index in kf.split(df_array):
        i = i + 1

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

        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train,model__sample_weight=sample_weights_train)

        validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
        validate(rf_model_pipeline,'rf','std',i,X_test,y_test)
        validate(svm_model_pipeline,'svm','std',i,X_test,y_test)

        validate(lr_fair_model_pipeline,'lr','fair',i,X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,'rf','fair',i,X_fair_test,y_fair_test)
        validate(svm_model_pipeline,'svm','fair',i,X_fair_test,y_fair_test)


    print(lr_model_pipeline.score(X,y))
    print(rf_model_pipeline.score(X,y))
    print(svm_model_pipeline.score(X,y))
    
    print(lr_fair_model_pipeline.score(X_fair,y_fair))
    print(rf_fair_model_pipeline.score(X_fair,y_fair))
    print(svm_model_pipeline.score(X_fair,y_fair))

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

    ed_df_aif360 = BinaryLabelDataset(
        df=df_mod,
        favorable_label=1,
        unfavorable_label=0,
        label_names = ['y'],
        protected_attribute_names=education_features
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

    return (ed_df_trans.convert_to_dataframe()[0],ed_df_trans.instance_weights)


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
    with open(f"./reports/fairness_reports/bank_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

load_dataset()