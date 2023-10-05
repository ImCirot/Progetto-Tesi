import numpy as np 
import pandas as pd 
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing


def load_dataset():
    ## funzione di load del dataset e drop features superflue

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # drop ID dal dataset
    df.drop('ID',inplace=True,axis=1)

    training_model(df)


def training_model(dataset):
    ## funzione di apprendimento del modello sul dataset

    # setting variabili protette
    protected_features_names = [
        'race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male'
    ]

    # creiamo un dataset fair effettuando modifiche al dataset originale
    test_fairness(dataset)

    # # setting nomi features del dataset
    # features = dataset.columns.tolist()

    # # rimuoviamo il nome della feature target dalla lista nomi features
    # features.remove('salary')

    # # setting nome target feature
    # target = ['salary']

    # # setting dataset features
    # X = dataset[features]

    # # setting dataset target feature
    # y = dataset[target]

    # # setting strategia KFold con 10 iterazioni standard
    # kf = KFold(n_splits=10)

    # # trasformiamo dataset in array per estrarre indici per la strategia KFold
    # df_array = np.asarray(dataset)

    # # setting contatore iterazioni KFold
    # i = 0

    # # costruiamo il modello standard tramite pipeline contenente uno scaler per la normalizzazione dati e un regressore
    # model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # # costruiamo un modello tramite pipeline su cui utilizzare un dataset opportunamente modificato per aumentare fairness
    # fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # # ciclo strategia KFold
    # for train_index,test_index in kf.split(df_array):
    #     i = i+1

    #     # setting training set per l'i-iterazione della strategia KFold
    #     X_train = X.iloc[train_index]
    #     y_train = y.iloc[train_index]

    #     # setting test set per l'i-iterazione della sstrategia KFold
    #     X_test = X.iloc[test_index]
    #     y_test = y.iloc[test_index]

    #     # training del modello sul training set dell'i-esima iterazione
    #     model_pipeline.fit(X_train,y_train.values.ravel())

    #     # calcolo metriche di valutazione sul modello dell'i-esima iterazione
    #     validate(model_pipeline,i,"std_models",X_test,y_test)


def validate(ml_model,index,model_type,X_test,y_test):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/{model_type}/aif360/adult_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_type}/aif360/adult_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write('\n')

def test_fairness(original_dataset):
    ## funzione che testa la fairness del dataset tramite libreria AIF360 e restituisce un dataset fair opportunamente modificato

    race_features = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "race"
    aif_race_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_features,
        privileged_protected_attributes=['race_White']
    )
    
    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui "bianchi" come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di razze diverse, che vengono settati come appartenenti al gruppo sfavorito.
    race_privileged_groups = [{'race_White': 1}]
    race_unprivileged_groups = [{'race_White': 0}]

    # Calcolo della metrica sul dataset originale
    race_metric_original = BinaryLabelDatasetMetric(dataset=aif_race_dataset, unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)    
    
    # è possibile visualizzare il numero di individui con predizione positiva appartenenti ad entrambi i gruppi tramite le linee di codice commentate di seguito
    # print(f'Num. of positive instances of priv_group before: {race_metric_original.num_positives(privileged=True)}')
    # print(f'Num. of positive instances of unpriv_group before: {race_metric_original.num_positives(privileged=False)}')

    # viene stampata a schermo la mean_difference 
    # (differenza fra predizioni positive di indivudi sfavoriti rispetto alle predizioni positive degli individui favoriti)
    print(f'Race metrics before: {race_metric_original.mean_difference()}')

    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    race_dataset_transformed = RACE_RW.fit_transform(aif_race_dataset)

    # è possibile visualizzare il numero di individui con predizione positiva appartenenti ad entrambi i gruppi dopo il processo di mitigazione attuato
    # nelle seguenti linee di codice commentate
    # print(f'Num. of positive instances of priv_group after: {race_metric_transformed.num_positives(privileged=True)}')
    # print(f'Num. of positive instances of unpriv_group after: {race_metric_transformed.num_positives(privileged=False)}')

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_dataset_transformed,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato
    print(f'Race metrics after: {race_metric_transformed.mean_difference()}')

    #
    #
    #
    # possiamo concludere che, in seguito alla ricalibrazione offerta, se il valore è diminuito allora siamo riusciti a rimuovere questa disparità nelle istanze 
    # di input con esito positivo, questo ci permette di assumere che un modello addestrato su questo nuovo dataset, in grado di fornire un numero maggiore di 
    # istanze positive anche per il gruppo sfavorito, possa portare a realizzare un modello più fair.
    #
    #
    #
    
    # setting nome varibili sensibili legate al sesso
    sex_features = ['sex_Male','sex_Female']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "sex"
    aif_sex_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male']
    )

    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui di sesso maschile come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di sesso femminile, che vengono settati come appartenenti al gruppo sfavorito.
    sex_privileged_groups = [{'sex_Male': 1}]
    sex_unprivileged_groups = [{'sex_Female': 1}]

    # Calcolo della metrica sul dataset originale
    sex_metric_original = BinaryLabelDatasetMetric(dataset=aif_sex_dataset, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups) 

    # è possibile visualizzare il numero di individui con predizione positiva appartenenti ad entrambi i gruppi tramite le linee di codice commentate di seguito
    # print(f'Num. of positive instances of priv_group before: {sex_metric_original.num_positives(privileged=True)}')
    # print(f'Num. of positive instances of unpriv_group before: {sex_metric_original.num_positives(privileged=False)}')
    
    # stampiamo la metrica mean_differe    
    print(f'Sex metrics before: {sex_metric_original.mean_difference()}')
    
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    SEX_RW = Reweighing(unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    sex_dataset_transformed = SEX_RW.fit_transform(aif_sex_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=sex_dataset_transformed,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)
    
    # è possibile visualizzare il numero di individui con predizione positiva appartenenti ad entrambi i gruppi dopo il processo di mitigazione attuato
    # nelle seguenti linee di codice commentate
    # print(f'Num. of positive instances of priv_group after: {sex_metric_transformed.num_positives(privileged=True)}')
    # print(f'Num. of positive instances of unpriv_group after: {sex_metric_transformed.num_positives(privileged=False)}')

    # stampa della mean_difference del nuovo modello bilanciato
    print(f'Sex metrics after: {sex_metric_transformed.mean_difference()}')

    #
    #
    #
    # eseguendo questa parte di codice notiamo come, per gli attributi legati al sesso, il numero di istanze di individui di sesso maschile è più grande
    # quindi è giusto trattarli come gruppo favorito, ma al momento del calcolo della metrica si può notare un valore negativo, il che indica che, nonostante
    # il numero di indivudi di sesso maschile superiore agli individui di sesso femminile, la differenza fra casi positivi gruppo non favorito e gruppo favorito
    # vada in favore del gruppo sfavorito.
    # Questo risulta in un bilanciamento incorretto che va ad aggiungere piuttosto che rimuovere la disparità passando da un valore che favoriva il gruppo
    # sfavorito in un valore che favorisce ampiamente il gruppo favorito, aumentando ancora di più la disparità fra i gruppi
    #
    #
    #

    # scegliamo ora di effettuare la stessa operazione, prendendo contemporaneamente sia gli attributi legati al sesso che alla razza e testandone le disparità
    # presenti nel dataset
    aif_overall_dataset = BinaryLabelDataset(

    )

    
    # aif_df = aif_race_dataset.convert_to_dataframe()[0]


load_dataset()