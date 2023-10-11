import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import *
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from fairlearn.postprocessing import ThresholdOptimizer,plot_threshold_optimizer
import pickle
import json

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # print di debug
    # print(df.head)

    training_model(df)

def training_model(dataset):
    ## funzione di sviluppo del modello

    # drop delle features superflue
    dataset.drop("ID",axis=1,inplace=True)

    # lista con tutte le features del dataset
    features = dataset.columns.tolist()

    # drop dalla lista del nome della variabile target
    features.remove('salary')

    # setting lista contenente nomi degli attributi protetti
    protected_features_names = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male']

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente valori attributi sensibili
    g = dataset[protected_features_names]

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))

    # setting pipeline da addestrare sul dataset soggetto ad operazioni di fairness
    lr_fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_fair_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_fair_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))


    # richiamiamo la funzione che dal dataset originale genera un nuovo dataset modificato rimuovendo la correlazione fra gli attributi sensibili e non
    # del dataset
    fair_dataset = fairness_preprocess_op(dataset=dataset,protected_features_names=protected_features_names)

    # estraiamo feature X ed y dal dataset ricalibrato
    X_fair = fair_dataset[features]
    y_fair = fair_dataset['salary']

    # setting della strategia KFold con standard di 10 gruppi
    kf = KFold(n_splits=10)

    # inizializzo contatore per il ciclo KFold
    i = 0

    # setting array contenente valori del dataframe
    df_array = np.asarray(dataset)

    # costruiamo un operatore di postprocessing per cercare di ottimizzare il modello 
    lr_treshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    rf_treshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    svm_treshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    # ciclo strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        # setting traing set X ed y dell'iterazione i-esima
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        g_train = g.iloc[train_index]

        # setting test set X ed y dell'iterazione i-esima
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        g_test = g.iloc[test_index]

        # addestriamo i modelli sui set della iterazione i-esima della KFold
        lr_model_pipeline.fit(X_train,y_train)
        rf_model_pipeline.fit(X_train,y_train)
        svm_model_pipeline.fit(X_train,y_train)

        # validiamo i modelli per ottenere metriche di valutazione
        validate(lr_model_pipeline,'std_models','lr', i, X_test, y_test)
        validate(rf_model_pipeline,'std_models','rf',i,X_test,y_test)
        validate(svm_model_pipeline,'std_models','svm',i,X_test,y_test)

        # addestriamo ora un modello sul dataframe in precedenza ricalibrato usando fairlearn

        # settiamo i training set sul dataset modificato
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        # settiamo i test set sul dataset modificato
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # addestriamo i modelli sull'iterazione i-esima della KFold
        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train)
        
        # validiamo i modelli ottenuti per fornire metriche di valutazione
        validate(lr_fair_model_pipeline,'fair_models','lr',i,X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,'fair_models','rf',i,X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,'fair_models','svm',i,X_fair_test,y_fair_test)

        lr_treshold.fit(X_train,y_train,sensitive_features=g_train)
        rf_treshold.fit(X_train,y_train,sensitive_features=g_train)
        svm_treshold.fit(X_train,y_train,sensitive_features=g_train)

        validate_postop(lr_treshold,"lr",i,X_test,y_test,g_test)
        validate_postop(rf_treshold,'rf',i,X_test,y_test,g_test)
        validate_postop(svm_treshold,'svm',i,X_test,y_test,g_test)

        # linea di codice per plottare il accuracy e selection_rate del modello con operazione di postop
        # plot_threshold_optimizer(lr_treshold)
    
    # per stampare i grafici generati
    plt.show()

    

    # salviamo i modelli ottenuti
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_fairlearn_adult_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_fairlearn_adult_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_fairlearn_adult_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_fairlearn_adult_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_fairlearn_adult_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_fairlearn_adult_model.sav','wb'))
    pickle.dump(lr_treshold,open('./output_models/postop_models/threshold_lr_fairlearn_adult_model.sav','wb'))
    pickle.dump(rf_treshold,open('./output_models/postop_models/threshold_rf_fairlearn_adult_model.sav','wb'))
    pickle.dump(svm_treshold,open('./output_models/postop_models/threshold_svm_fairlearn_adult_model.sav','wb'))


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
    show_correlation_heatmap(dataset=dataset,title='original dataset')
    show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset


def show_correlation_heatmap(dataset,title):
    ## funzione che genera heatmap sul dataset fornito usando attributi sensibili e non per mostrare il grado di correlazione presente

    # settiamo una lista contenente alcuni attrubi del dataset e le variabili sensibili
    sens_features_and_unses_features = ['sex_Male','sex_Female','race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','age','education-num','hours-per-week','salary','workclass_Federal-gov']
    
    # creiamo una heatmap del dataset che mostra la correlazione fra gli attributi dichiarati in precedenza
    sns.heatmap(dataset[sens_features_and_unses_features].corr(),annot=True,cmap='coolwarm',fmt='.2f',)
    plt.title(title)
    plt.show()

def validate(ml_model, model_vers, model_type, index, X_test, y_test):
    ## funzione utile a calcolare metriche del modello realizzato

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
    with open(f"./reports/{model_vers}/fairlearn/adult/{model_type}_adult_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/{model_vers}/fairlearn/adult/{model_type}_adult_metrics_report.txt",open_type) as f:
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
    with open(f"./reports/postop_models/{model_type}_adult_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with open(f"./reports/postop_models/{model_type}_adult_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

load_dataset()