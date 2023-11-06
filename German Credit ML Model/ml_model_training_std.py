import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime

@track_emissions(offline=True, country_iso_code="ITA")
def traning_and_testing_model():
    ## Funzione per il training e testing del modello scelto

    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    df['Target'] = df['Target'].replace(2,0)

    features = df.columns.tolist()
    features.remove('Target')

    target = ['Target']

    X = df[features]
    y = df[target]

    # Si crea un array del dataframe utile per la KFold
    df_array = np.array(df)

    # Settiamo il numero di gruppi della strategia KFold a 10
    kf = KFold(n_splits=10)

    # inizializiamo contatore i
    i = 0

    # Creiamo due pipeline che effettuano delle ulteriori operazioni di scaling dei dati per addestriare il modello
    # in particolare la pipeline standard sarà addestrata sui dati as-is
    # mentre la fair pipeline verrà addestrata su dati sui vengono applicate strategie di fairness
    # volte a rimuovere discriminazione e bias nel dataset di training
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    # Strategia KFold
    for train_index, test_index in kf.split(df_array):
        
        i = i+1
        
        print(f'\n######### Inizio {i} iterazione #########\n')

        # setting del training set dell'i-esima iterazione 
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]

        # setting del test set dell'i-esima iterazione 
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]

        # fit del modello sul training set dell'i-esima iterazione
        lr_model_pipeline.fit(X_train,y_train.values.ravel())
        rf_model_pipeline.fit(X_train,y_train.values.ravel())
        svm_model_pipeline.fit(X_train,y_train.values.ravel())
        xgb_model_pipeline.fit(X_train,y_train.values.ravel())

        # Stampiamo metriche di valutazione per il modello
        validate(lr_model_pipeline, i, "std_models", 'lr', X_test, y_test)
        validate(rf_model_pipeline,i,'std_models','rf',X_test,y_test)
        validate(svm_model_pipeline,i,'std_models','svm',X_test,y_test)
        validate(xgb_model_pipeline,i,'std_models','xgb',X_test,y_test)

        print(f'\n######### Fine {i} iterazione #########\n')

    print(f'######### Inizio stesura report finale #########')
    with open('./reports/final_scores/std/credit_scores.txt','w') as f:
        f.write(f'LR std model: {str(lr_model_pipeline.score(X,y))}\n')
        f.write(f'RF std model: {str(rf_model_pipeline.score(X,y))}\n')
        f.write(f'SVM std model: {str(svm_model_pipeline.score(X,y))}\n')
        f.write(f'XGB std model: {str(xgb_model_pipeline.score(X,y))}\n')
    
    print(f'######### Inizio salvataggio modelli #########')
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_credit_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_credit_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_credit_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_credit_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')
            
def validate(ml_model,index,model_vers,model_type,X_test,y_test):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/{model_vers}/credit/{model_type}_credit_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/credit/{model_type}_credit_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def print_time(time):
    with open('./reports/time_reports/std/credit_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

# Chiamata funzione inizale di training e testing
start = datetime.now()
traning_and_testing_model()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)