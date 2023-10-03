import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import *
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # print di debug
    print(df.head)

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
    protected_features_names = ['race','sex']

    # settiamo delle metriche utili per poter fornire delle valutazioni sugli attributi sensibili tramite il framework FairLearn
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "selection rate": selection_rate,
        "count": count,
    }

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente le features protette
    protected_features = dataset[protected_features_names]

    # setting del set contenente il sesso degli individui presenti nel dataset
    sex = dataset['sex']

    # setting del set contenente razza degli indivuidi presenti nel dataset
    race = dataset['race']

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())


    # setting della strategia KFold con standard di 10 gruppi
    kf = KFold(n_splits=10)

    # inizializzo contatore per il ciclo KFold
    i = 0

    # setting array contenente valori del dataframe
    df_array = np.asarray(dataset)

    # ciclo strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        # setting traing set X ed y dell'iterazione i-esima
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # setting training set delle sole varibili protette dell'iterazione i-esima
        protected_features_train = protected_features.iloc[train_index]

        # setting training set della singole variabili protette contenenti informazioni sul sesso e razza dell'individuo
        sex_train = sex.iloc[train_index]
        race_train = race.iloc[train_index]

        # setting test set X ed y dell'iterazione i-esima
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # setting training set delle sole varibili protette dell'iterazione i-esima
        protected_features_test = protected_features.iloc[test_index]

        # setting test set della singole variabili protette contenenti informazioni sul sesso e razza dell'individuo
        sex_test = sex.iloc[test_index]
        race_test = race.iloc[test_index]

        # training modello sul set X ed y dell'iterazione i-esima
        model_pipeline.fit(X_train,y_train)

        # produciamo una predizione di test per l'iterazione i-esima
        pred = model_pipeline.predict(X_test)

        # calcoliamo delle metriche di fairness sulla base degli attributi sensibili
        # overall_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=protected_features_test)
        # mf.by_group.plot.bar(
        #     subplots=True,
        #     layout=[3, 3],
        #     legend=False,
        #     figsize=[12, 8],
        #     title="Show all metrics",
        # )

        # calcoiamo delle metriche di fairness sulla base dell'attributo protetto "sex"
        sex_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=sex_test)
        sex_mf.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Show all metrics",
        )

        # calcoiamo delle metriche di fairness sulla base dell'attributo protetto "race"
        race_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=race_test)
        race_mf.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Show all metrics",
        )

        validate(model_pipeline, i, X_test, y_test)
    
    # per stampare i grafici generati
    plt.show()

def validate(ml_model, index, X_test, y_test):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/quality_reports/adult_fairlearn_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/quality_reports/adult_fairlearn_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')



load_dataset()