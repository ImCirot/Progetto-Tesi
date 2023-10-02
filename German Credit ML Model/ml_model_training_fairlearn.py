import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt
from fairlearn.reductions import *

def training_model(dataset):
    ## funzione che addestra il modello sul dataset utilizzando strategia KFold

    # trasformiamo dataset in array per usare indici strategia KFold
    df_array = np.asarray(dataset)

    # evidenziamo le features utili alla predizione
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista features la feature target
    features.remove('Target')

    # evidenziamo gli attributi sensibili del dataset
    sex_features = [
        'sex_A91','sex_A92','sex_A93','sex_A94'
    ]

    # settiamo delle metriche utili per poter fornire delle valutazioni sugli attributi sensibili tramite il framework FairLearn
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "selection rate": selection_rate,
        "count": count,
    }

    # settiamo la nostra X sulle sole variabili di features
    X = dataset[features]

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    dataset['Target'] = dataset['Target'].replace(2,0)

    # settiamo la nostra y sulla variabile da predire
    y = dataset['Target']

    # settiamo un dataframe contenente solamente i valori degli attributi sensibili (utile per utilizzare il framework FairLearn)
    sex = dataset[sex_features]

    # settiamo contatore per ciclo KFold
    i = 0

    # settiamo il numero di ripetizioni uguale a 10, standard per la strategia KFold
    kf = KFold(n_splits=10)

    # Creiamo una pipeline contenente il modello basato su regressione logistica e uno scaler per poter scalare i dati correttamente per poter
    # utilizzare correttamente il modello
    model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    for train_index, test_index in kf.split(df_array):
        i = i + 1

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        sex_train = sex.loc[train_index]

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        sex_test = sex.loc[test_index]

        # fitting del modello sui dati di training per l'iterazione i-esima
        model_pipeline.fit(X_train,y_train)

        # produciamo una predizione di test per l'iterazione i-esima
        pred = model_pipeline.predict(X_test)

        # calcoliamo delle metriche di fairness sulla base degli attributi sensibili
        mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=sex_test)
        # fnr = false_negative_rate(y_true=y_test,y_pred=pred,pos_label=1)
        # fpr = false_positive_rate(y_true=y_test,y_pred=pred,pos_label=1)
        # print(fnr)
        # print(fpr)  
        mf.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Show all metrics",
        )

        # per mostrare i grafici generati per ogni iterazione
        # plt.show()

        # proviamo alcune operazioni di postprocessing sul modello prodotto
        postprocess_model = ThresholdOptimizer(
            estimator=model_pipeline,
            constraints='equalized_odds',
            objective='balanced_accuracy_score',
            prefit=True,
            predict_method='predict_proba'
        )

        postprocess_model.fit(X_train,y_train,sensitive_features=sex_train)
        fair_pred = postprocess_model.predict(X_test,sensitive_features=sex_test)
        fair_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=fair_pred,sensitive_features=sex_test)
        mf.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Show all metrics",
        )
        

        # validiamo i risultati prodotti dal modello all'iterazione i-esima chiamando una funzione che realizza metriche di valutazione
        validate(model_pipeline, i, X_test, y_test)

    # per mostrare grafici
    # plt.show()


def validate(ml_model,index,X_test,y_test):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/quality_reports/credit_fairlearn_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/quality_reports/credit_fairlearn_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./German Credit Dataset/dataset_modificato.csv')

    df.drop('ID', axis=1, inplace=True)

    training_model(df)


load_dataset()