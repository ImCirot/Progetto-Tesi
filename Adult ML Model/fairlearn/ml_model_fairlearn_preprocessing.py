import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import MetricFrame,demographic_parity_difference,demographic_parity_ratio,equalized_odds_difference
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import xgboost as xgb
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')
    
    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione di sviluppo del modello

    dataset = dataset.drop('ID',axis=1)
    
    # lista con tutte le features del dataset
    features = dataset.columns.tolist()

    # drop dalla lista del nome della variabile target
    features.remove('salary')

    # setting lista contenente nomi degli attributi protetti
    protected_features_names = ['sex_Female','sex_Male','age']
    sex_features_names = ['sex_Female','sex_Male']
    age_feature = ['age']

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente valori attributi sensibili
    g= dataset[protected_features_names]

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_adult_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_adult_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_adult_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_adult_model.sav','rb'))

    # setting pipeline da addestrare sul dataset soggetto ad operazioni di fairness
    lr_fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_fair_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_fair_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_fair_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier(objective='binary:logistic', random_state=42))


    # richiamiamo la funzione che dal dataset originale genera un nuovo dataset modificato rimuovendo la correlazione fra gli attributi sensibili e non
    # del dataset
    fair_dataset = fairness_preprocess_op(dataset=dataset,protected_features_names=protected_features_names)

    # estraiamo feature X ed y dal dataset ricalibrato
    X_fair = fair_dataset[features]
    y_fair = fair_dataset['salary']

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)
    X_fair_train, X_fair_test, y_fair_train, y_fair_test = train_test_split(X_fair,y_fair,test_size=0.2,random_state=42)

    # addestriamo i modelli
    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train)
    
    # validiamo i modelli ottenuti per fornire metriche di valutazione
    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    print(f'######### Testing Fairness #########')
    lr_std_pred = lr_model_pipeline.predict(X_test)
    lr_fair_pred = lr_fair_model_pipeline.predict(X_test)

    rf_std_pred = rf_model_pipeline.predict(X_test)
    rf_fair_pred = rf_fair_model_pipeline.predict(X_test)

    svm_std_pred = svm_model_pipeline.predict(X_test)
    svm_fair_pred = svm_fair_model_pipeline.predict(X_test)

    xgb_std_pred = xgb_model_pipeline.predict(X_test)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X_test)

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

        sex_DI_value = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])
        sex_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])
        sex_eq_odds_diff = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_features_names])

        age_DI_value = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_eq_odds_diff = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/preprocessing/fairlearn/adult_report.txt',open_type) as f:
            f.write(f'{name} sex DI: {round(sex_DI_value,3)}\n')
            f.write(f'{name} sex mean_diff: {round(sex_mean_diff,3)}\n')
            f.write(f'{name} sex eq_odds_diff: {round(sex_eq_odds_diff,3)}\n')
            f.write(f'{name} age DI: {round(age_DI_value,3)}\n')
            f.write(f'{name} age mean_diff: {round(age_mean_diff,3)}\n')
            f.write(f'{name} age eq_odds_diff: {round(age_eq_odds_diff,3)}\n')
    
    # salviamo i modelli ottenuti
    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_fairlearn_adult_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_fairlearn_adult_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_fairlearn_adult_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_fairlearn_adult_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def fairness_preprocess_op(dataset, protected_features_names):
    ## funzione che utilizza classe offerta da fairlearn in grado di mitigare la correlazione fra gli attributi sensibili e non del dataset

    # lista contenente nomi colonne del dataset escluse le feature sensibili
    features_names = dataset.columns.tolist()
    for feature in protected_features_names:
        features_names.remove(feature)

    # creiamo un oggetto fornito dalla libreria Fairlean in grado di rimuovere correlazione fra le feature del dataset e le feature sensibili
    corr_remover = CorrelationRemover(sensitive_feature_ids=protected_features_names,alpha=1.0)
    fair_dataset = corr_remover.fit_transform(dataset)

    # ricostruiamo il dataframe inserendo le feature appena modificate
    fair_dataset = pd.DataFrame(
        fair_dataset, columns=features_names
    )

    # inseriamo nel nuovo dataframe le variabili sensibili rimosse in precedenza e la variabile target
    fair_dataset[protected_features_names] = dataset[protected_features_names]
    fair_dataset['salary'] = dataset['salary']

    # stampiamo heatmap che confrontano il grado di correlazione fra gli attributi sensibili e alcuni attributi del dataset
    # show_correlation_heatmap(dataset=dataset,title='original dataset')
    # show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset


def show_correlation_heatmap(dataset,title):
    ## funzione che genera heatmap sul dataset fornito usando attributi sensibili e non per mostrare il grado di correlazione presente

    # settiamo una lista contenente alcuni attrubi del dataset e le variabili sensibili
    sens_features_and_unses_features = ['sex_Male','sex_Female','race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','age','education-num','hours-per-week','salary','workclass_Federal-gov']
    
    # creiamo una heatmap del dataset che mostra la correlazione fra gli attributi dichiarati in precedenza
    sns.heatmap(dataset[sens_features_and_unses_features].corr(),annot=True,cmap='coolwarm',fmt='.2f',)
    plt.title(title)
    plt.show()

def validate(ml_model, model_type, X_test, y_test, first=False):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    f1 = f1_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    

    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/preprocessing_models/fairlearn/adult_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nF1 score: {round(f1,3)}\n')
        f.write(f"Precision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/fairlearn/adult_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()