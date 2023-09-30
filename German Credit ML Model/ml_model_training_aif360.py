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
from aif360.datasets import BinaryLabelDataset


@track_emissions(offline=True, country_iso_code="ITA")
def traning_and_testing_model():
    ## Funzione per il training e testing del modello scelto

    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # print di debug
    # pd.options.display.max_columns = 2
    # print(df.head())

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
    fair_pipe = make_pipeline(StandardScaler(), LogisticRegression())
    standard_pipe = make_pipeline(StandardScaler(), LogisticRegression())

    # Strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        split_dataset_train = df.loc[train_index]
        
        fair_dataset = test_fairness(split_dataset_train, i)

        # i due sottoinsiemi di training su cui è stata effettuata fairness
        X_train_fair = fair_dataset[features]
        y_train_fair = fair_dataset[target]

        # i due sottoinsiemi di dati senza alcuna modifica
        X_train_std = X.loc[train_index]
        y_train_std = y.loc[train_index]

        X_test = X.loc[test_index]
        y_test = y.loc[test_index]


        # Fit dei dati sul nostro modello fair tramite il gruppo di training attuale
        fair_pipe.fit(X_train_fair,y_train_fair.values.ravel())

        # Fit dei dati sul nostro modello standard traminte il gruppo di training standard attuale 
        standard_pipe.fit(X_train_std,y_train_std.values.ravel())

        # Stampiamo metriche di valutazione per entrambi i modelli generati
        validate(fair_pipe, "fair", i, X_test,y_test.values.ravel())
        validate(standard_pipe, "standard", i, X_test, y_test.values.ravel())

    # # Test di predizione del modello
    # prediction = pd.read_csv('./prediction.csv')
    # print(f"My prediction: {pipe.predict(prediction)}")



def validate(ml_model, type, index, X_test, y_test):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/quality_reports/{type}_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/quality_reports/{type}_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

def test_fairness(dataset, index):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    # Attributi sensibili
    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    # Attributi sensibili con vantaggio
    privileged_attribute_names = [
        'sex_A91','sex_A92','sex_A94'
    ]

    # Setting del dataset per l'utilizzo dell'API AIF360
    # dataset_origin_train = StandardDataset(
    #     df=dataset,
    #     label_name='Target',
    #     favorable_classes=[1],
    #     protected_attribute_names=protected_attribute_names,
    #     privileged_classes=[lambda x: x == 1]
    # )

    dataset_origin_train = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=2,
        label_names=['Target'],
        protected_attribute_names=protected_attribute_names,
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex_A93': 1}]
    unprivileged_groups = [{'sex_A93': 0}]

    # Calcolo della metrica sul dataset originale
    metric_original_train = BinaryLabelDatasetMetric(dataset=dataset_origin_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    
    # Se la metrica originale ottiene già valore 0.0, allora il dataset è gia equo e non ha bisogno di ulteriori operazioni
    if(metric_original_train.mean_difference() != 0.0):
        # Utilizzamo un operatore di bilanciamento offerto dall'API AIF360
        RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

        # Bilanciamo il dataset
        dataset_transformed_train = RW.fit_transform(dataset_origin_train)
        # Ricalcoliamo la metrica
        metric_transformed_train = BinaryLabelDatasetMetric(dataset=dataset_transformed_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # Creiamo un nuovo dataframe sulla base del modello ripesato dall'operazione precedente
    fair_dataset = dataset_transformed_train.convert_to_dataframe()[0]

    # Chiamata alla funzione per generare un report dei valori ottenuti
    print_fairness_metrics(metric_original_train.mean_difference(), metric_transformed_train.mean_difference(), "mean_difference", index)

    return fair_dataset
    
def print_fairness_metrics(original_metric, transformed_metric, metric_type, index):
    ## funzione per creare file di report di metriche di fairness

    # Scegliamo il tipo apertura file, se è la prima iteraz. creiamo file
    if index == 1:
        open_mode = 'w'
    else:
        open_mode = 'a'
    
    # Creiamo un file nella cartella reports con lo stesso nome della metrica scelta
    with open(f'./reports/fairness_reports/{metric_type}_report.txt', open_mode) as f:
        f.write(f'{metric_type}: iteration {index}\n')
        f.write(f'Original metric: {original_metric}\n')
        f.write(f'Metric after: {transformed_metric}\n')
        f.write('\n')


# Chiamata funzione inizale di training e testing
traning_and_testing_model()