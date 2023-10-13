from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
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

    X = df[feature_names]
    
    y = df['y']
    
    g = df[protected_features_names]

    fair_dataset = fairness_preprocess_op(dataset=df, protected_features_names=protected_features_names)

    # lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    # rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    # svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))

    # i = 0

    # for train_index, test_index in kf.split(df_array):
    #     i = i + 1

    #     X_train = X.iloc[train_index]
    #     y_train = y.iloc[train_index]

    #     X_test = X.iloc[test_index]
    #     y_test = y.iloc[test_index]

    #     lr_model_pipeline.fit(X_train,y_train)
    #     rf_model_pipeline.fit(X_train,y_train)
    #     svm_model_pipeline.fit(X_train,y_train)

    #     validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
    #     validate(rf_model_pipeline,'rf','std',i,X_test,y_test)
    #     validate(svm_model_pipeline,'svm','std',i,X_test,y_test)


    # print(lr_model_pipeline.score(X,y))
    # print(rf_model_pipeline.score(X,y))
    # print(svm_model_pipeline.score(X,y))

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

    with  open(f"./reports/{ml_vers}_models/fairlearn/bank/{ml_type}_marketing_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')

    with  open(f"./reports/{ml_vers}_models/fairlearn/bank/{ml_type}_marketing_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

load_dataset()