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

    # Creiamo una pipeline che effettua delle ulteriori operazioni di scaling dei dati per addestriare il modello
    pipe = make_pipeline(StandardScaler(), LogisticRegression())

    # Strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        split_dataset_train = df.loc[train_index]

        fair_dataset = test_fairness(split_dataset_train)

        X_train = fair_dataset[features]

        y_train = fair_dataset[target]

        X_test = X.loc[test_index]
        y_test = y.loc[test_index]

        # Fit dei dati sul nostro modello tramite il gruppo di training attuale
        pipe.fit(X_train,y_train.values.ravel())

        # Stampiamo metriche di valutazione
        validate(pipe, i, X_test,y_test.values.ravel())

    # # Test di predizione del modello
    # prediction = pd.read_csv('./prediction.csv')
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
    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    dataset_origin_train = StandardDataset(
        df=dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=protected_attribute_names,
        privileged_classes=[lambda x: x == 1]
    )

    privileged_groups = [{'sex_A91': 1}]
    unprivileged_groups = [{'sex_A91': 0}]

    metric_original_train = BinaryLabelDatasetMetric(dataset=dataset_origin_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    
    print(f'Differenze di output fra gruppo privilegiato e non privilegiato nel database originale: {metric_original_train.mean_difference()}')

    if(metric_original_train.mean_difference() != 0.0):
        RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

        dataset_transformed_train = RW.fit_transform(dataset_origin_train)

        metric_transformed_train = BinaryLabelDatasetMetric(dataset=dataset_transformed_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        print(f'Differenze di output fra gruppo privilegiato e non privilegiato nel database pesato: {metric_transformed_train.mean_difference()}')

    (fair_dataset,dist) = dataset_transformed_train.convert_to_dataframe()

    return fair_dataset
    

# Chiamata funzione inizale di training e testing
traning_and_testing_model()