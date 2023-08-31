import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset

@track_emissions(offline=True, country_iso_code="ITA")
def traning_and_testing_model():
    ## Funzione per il training e testing del modello scelto

    df = pd.read_csv("./Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # print di debug
    # pd.options.display.max_columns = 2
    # print(df.head())

    # Setting delle feature e del target
    features = [
        'Status of exisiting checking account',
        'Duration in month',
        'Credit history',
        'Purpose',
        'Credit amount',
        'Savings account/bonds',
        'Present employment since',
        'Installment rate in percentage of disposable income',
        'Sex_0',
        'Sex_1',
        'Sex_2',
        'Sex_3',
        'Other debtors / guarantors',
        'Present residence since',
        'Property',
        'Age in years',
        'Other installment plans',
        'Housing',
        'Number of existing credits at this bank',
        'Job',
        'Number of people being liable to provide maintenance for',
        'Telephone',
        'foreign worker',
    ]

    target = ['Target']

    X = df[features]

    y = df[target]

    g = {'Sex_0','Sex_1','Sex_2','Sex_3'}

    dataset = StandardDataset(
        df=df,
        label_name='Target',
        protected_attribute_names=g
    )
    
    # Testiamo la fairness dei gruppi appena ottenuti
    test_fairness(dataset)

    # Si crea un array del dataframe utile per la KFold
    df_array = np.array(df)

    # Settiamo il numero di gruppi della strategia KFold a 10
    kf = KFold(n_splits=10)

    # inizializiamo contatore i
    i = 0

    # Creiamo una pipeline che effettua delle ulteriori operazioni di scaling dei dati per addestriare il modello
    pipe = make_pipeline(StandardScaler(), LogisticRegression())

    # Strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # # Fit dei dati sul nostro modello tramite il gruppo di training attuale
        # pipe.fit(X_train,y_train.values.ravel())

        # # Stampiamo metriche di valutazione
        # validate(pipe, i, X_test,y_test.values.ravel())

    # # Test di predizione del modello
    # prediction = pd.read_csv('./prediction.csv')
    # prediction.drop('ID', inplace=True, axis=1)
    # print(f"My prediction: {pipe.predict(prediction)}")



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
    with open(f"./reports/matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

def test_fairness(dataset):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    # Attributi sensibili
    g = {'Sex_0','Sex_1','Sex_2', 'Sex_3'}

    metric_original_set = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'Sex_1': 0}], privileged_groups=[{'Sex_1':1}])
    print(metric_original_set)
    

# Chiamata funzione inizale di training e testing
traning_and_testing_model()