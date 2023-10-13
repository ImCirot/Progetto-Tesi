from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd

def load_dataset():
    df = pd.read_csv('./Bank Marketing Dataset/dataset.csv')

    kf = KFold(n_splits=10)

    df_array = np.asarray(df)

    feature_names = df.columns.tolist()

    feature_names.remove('y')

    X = df[feature_names]
    y = df['y']

    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))

    i = 0

    for train_index, test_index in kf.split(df_array):
        i = i + 1

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        lr_model_pipeline.fit(X_train,y_train)
        rf_model_pipeline.fit(X_train,y_train)

        validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
        validate(rf_model_pipeline,'rf','std',i,X_test,y_test)


    print(lr_model_pipeline.score(X,y))
    print(rf_model_pipeline.score(X,y))

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

load_dataset()