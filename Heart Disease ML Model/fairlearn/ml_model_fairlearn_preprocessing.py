import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fairlearn.preprocessing import CorrelationRemover
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from codecarbon import track_emissions
import xgboost as xgb
from datetime import datetime
from time import sleep


def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./Heart Disease Dataset/dataset.csv')

    df.drop('ID', axis=1, inplace=True)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(120)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione che addestra e valuta i modelli sulla base di un dataset fair

    # otteniamo la lista delle features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista la feature target
    features.remove("num")

    # settiamo il nome delle features sensibili
    protected_features = [
        'age','sex'
    ]

    sex_feature = ['sex']

    age_feature = ['age']

    # setting delle variabili X,y e la variabile g degli attributi sensibili
    X = dataset[features]
    y = dataset['num']
    g = dataset[protected_features]

    # carichiamo i modelli standard per effettuare dei confronti di fairness post training e testing dei modelli fair
    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_heart_disease_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_heart_disease_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_heart_disease_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_heart_disease_model.sav','rb'))

    # creiamo un nuovo modello da addestrare sul dataset modificato
    lr_fair_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf_fair_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_fair_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_fair_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    # otteniamo il dataset modificato sulla base di fairness
    fair_dataset = fairness_preprocess_op(dataset,protected_features)

    # otteniamo la lista delle features e rimuoviamo la variabile target dalla lista
    features_list = fair_dataset.columns.tolist()
    features_list.remove('num')

    # settiamo le variabili fair X e y
    X_fair = fair_dataset[features_list]
    y_fair = fair_dataset['num']
    
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X,y,g,test_size=0.2,random_state=42)
    
    # settiamo i subset di training e testing del dataset fair
    X_fair_train, X_fair_test, y_fair_train, y_fair_test = train_test_split(X_fair,y_fair,test_size=0.2,random_state=42)

    # addestriamo i modelli sul dataset fair
    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())

    # validiamo i risultati prodotti dai modelli
    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    # testiamo la fairness dei modelli ottenuti confrontando i modelli standard e fair
    # sulla base dei risultati prodotti dalla predizione dell'intero set X
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

    # calcoliamo il DI di ogni modello e stampiamo su un file i risultati
    for name,prediction in predictions.items():

        sex_DI_score = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        sex_eqodds = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        sex_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[sex_feature])
        age_DI_score = demographic_parity_ratio(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_eqodds = equalized_odds_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])
        age_mean_diff = demographic_parity_difference(y_true=y_test,y_pred=prediction,sensitive_features=g_test[age_feature])


        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/preprocessing/fairlearn/heart_disease_report.txt',open_type) as f:
            f.write(f'{name} Sex DI: {round(sex_DI_score,3)}\n')
            f.write(f'{name} Sex mean_diff: {round(sex_mean_diff,3)}\n')
            f.write(f'{name} Sex eq_odds_diff: {round(sex_eqodds,3)}\n')
            f.write(f'{name} Age DI: {round(age_DI_score,3)}\n')
            f.write(f'{name} Age mean_diff: {round(age_mean_diff,3)}\n')
            f.write(f'{name} Age eq_odds_diff: {round(age_eqodds,3)}\n')

    # salviamo i modelli
    print(f'######### Inizio salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_fairlearn_heart_disease_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_fairlearn_heart_disease_model.sav','wb'))
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    accuracy = accuracy_score(y_test,pred)

    f1 = f1_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/preprocessing_models/fairlearn/heart_disease_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nF1 score: {round(f1,3)}')
        f.write(f"\nPrecision: {round(precision,3)}")
        f.write(f'\nRecall: {round(recall,3)}\n')
        f.write('\n')


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
    fair_dataset['num'] = dataset['num']

    fair_dataset = fair_dataset[dataset.columns.tolist()]

    # stampiamo heatmap che confrontano il grado di correlazione fra gli attributi sensibili e alcuni attributi del dataset
    # show_correlation_heatmap(dataset=dataset,title='original dataset')
    # show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset

def show_correlation_heatmap(dataset,title):
    ## funzione che genera heatmap sul dataset fornito usando attributi sensibili e non per mostrare il grado di correlazione presente
    
    # creiamo una heatmap del dataset che mostra la correlazione fra gli attributi dichiarati in precedenza
    sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm',fmt='.2f',)
    plt.title(title)
    plt.show()

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/fairlearn/heart_disease_preprocessing_report.txt',open_type) as f:
        f.write(f'Elapsed time: {time} seconds.\n')

load_dataset()
