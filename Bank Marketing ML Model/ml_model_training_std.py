from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from codecarbon import track_emissions
import pickle
import xgboost as xgb
from datetime import datetime

@track_emissions(country_iso_code='ITA',offline=True)
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
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    i = 0

    for train_index, test_index in kf.split(df_array):
        i = i + 1

        print(f'\n######### Inizio {i} iterazione #########\n')
        
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        lr_model_pipeline.fit(X_train,y_train)
        rf_model_pipeline.fit(X_train,y_train)
        svm_model_pipeline.fit(X_train,y_train)
        xgb_model_pipeline.fit(X_train,y_train)

        validate(lr_model_pipeline,'lr','std',i,X_test,y_test)
        validate(rf_model_pipeline,'rf','std',i,X_test,y_test)
        validate(svm_model_pipeline,'svm','std',i,X_test,y_test)
        validate(xgb_model_pipeline,'xgb','std',i,X_test,y_test)

        print(f'\n######### Fine {i} iterazione #########\n')

    print(f'######### Inizio stesura report finale #########')
    with open('./reports/final_scores/std/bank_scores.txt','w') as f:
        f.write(f'LR std model: {str(lr_model_pipeline.score(X,y))}\n')
        f.write(f'RF std model: {str(rf_model_pipeline.score(X,y))}\n')
        f.write(f'SVM std model: {str(svm_model_pipeline.score(X,y))}\n')
        f.write(f'XGB std model: {str(xgb_model_pipeline.score(X,y))}\n')

    print(f'######### Inizio salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_bank_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_bank_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_bank_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_bank_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')


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

    with  open(f"./reports/{ml_vers}_models/bank/{ml_type}_marketing_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')

    with  open(f"./reports/{ml_vers}_models/bank/{ml_type}_marketing_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/std/bank_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')


start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)