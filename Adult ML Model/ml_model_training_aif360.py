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
from codecarbon import track_emissions
import pickle

@track_emissions(country_iso_code='ITA',offline=True)
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
    fair_dataset = test_fairness(dataset)

    # setting nomi features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo il nome della feature target dalla lista nomi features
    features.remove('salary')

    # setting nome target feature
    target = ['salary']

    # setting dataset features
    X = dataset[features]
    X_fair = fair_dataset[features]

    # setting dataset target feature
    y = dataset[target]
    y_fair = fair_dataset[target]

    # setting strategia KFold con 10 iterazioni standard
    kf = KFold(n_splits=10)

    # trasformiamo dataset in array per estrarre indici per la strategia KFold
    df_array = np.asarray(dataset)

    # setting contatore iterazioni KFold
    i = 0

    # costruiamo il modello standard tramite pipeline contenente uno scaler per la normalizzazione dati e un regressore
    model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # costruiamo un modello tramite pipeline su cui utilizzare un dataset opportunamente modificato per aumentare fairness
    fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # ciclo strategia KFold per il modello base
    for train_index,test_index in kf.split(df_array):
        i = i+1

        # setting training set per l'i-iterazione della strategia KFold 
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # setting test set per l'i-iterazione della sstrategia KFold 
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # training del modello base sul training set dell'i-esima iterazione
        model_pipeline.fit(X_train,y_train.values.ravel())

        # calcolo metriche di valutazione sul modello base dell'i-esima iterazione 
        validate(model_pipeline,i,"std_models",X_test,y_test)

    # trasformiamo dataframe in array per poter utilizzare la strategia KFold
    df_fair_array = np.asarray(fair_dataset)

    # resettiamo contatore i
    i = 0
    for train_index, test_index in kf.split(df_fair_array):
        i = i+1

        # setting training set per l'i-esima iterazione della strategia KFold per il modello fair
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        # setting test set per l'i-esima iterazione della strategia KFold per il modello fair
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # training del modello sul training set dell'i-esima iterazione
        fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())

        # calcolo metriche di valutazione sul modello fair dell'i-esima iterazione
        validate(fair_model_pipeline,i,'fair_models',X_fair_test,y_fair_test)

    pickle.dump(model_pipeline,open('./output_models/std_models/aif360_adult_model.sav','wb'))
    pickle.dump(fair_model_pipeline,open('./output_models/fair_models/aif360_adult_model.sav','wb'))

def validate(ml_model,index,model_type,X_test,y_test):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

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
        f.write(f'\nAUC ROC score: {auc_score}\n')
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
    
    # stampiamo la metrica mean_difference sul file di report    
    # (differenza fra predizioni positive di indivudi sfavoriti rispetto alle predizioni positive degli individui favoriti)
    print_fairness_metrics('mean_difference',race_metric_original.mean_difference(),'Race mean_difference before',first_message=True)

    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    race_dataset_transformed = RACE_RW.fit_transform(aif_race_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_dataset_transformed,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics('mean_difference',race_metric_transformed.mean_difference(),'Race mean_difference after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics('mean_difference',race_metric_original.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group before')
    print_fairness_metrics('mean_difference',race_metric_original.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics('mean_difference',race_metric_transformed.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group after')
    print_fairness_metrics('mean_difference',race_metric_transformed.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group after')

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
    
    # stampiamo la metrica mean_difference sul file di report    
    print_fairness_metrics('mean_difference',sex_metric_original.mean_difference(),'Sex mean_difference before')
    
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    SEX_RW = Reweighing(unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    sex_dataset_transformed = SEX_RW.fit_transform(aif_sex_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=sex_dataset_transformed,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics('mean_difference',sex_metric_transformed.mean_difference(),'Sex mean_difference value after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics('mean_difference',sex_metric_original.num_positives(privileged=True),'(SEX) Num. of positive instances of priv_group before')
    print_fairness_metrics('mean_difference',sex_metric_original.num_positives(privileged=False),'(SEX) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics('mean_difference',sex_metric_transformed.num_positives(privileged=True),'(SEX) Num. of positive instances of priv_group after')
    print_fairness_metrics('mean_difference',sex_metric_transformed.num_positives(privileged=False),'(SEX) Num. of positive instances of unpriv_group after')

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

    # lista dei nomi delle features protette
    protected_features = [
        'race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Male','sex_Female'
    ]

    # lista nomi delle features protette favorite
    privileged_features = [
        'sex_Male','race_White'
    ]

    aif_overall_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=protected_features,
        privileged_protected_attributes=privileged_features
    )

    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui bianchi di sesso maschile come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui altri individui indipendentemente dal sesso e razza.
    overall_privileged_groups = [{'sex_Male': 1, 'race_White': 1}]
    overall_unprivileged_groups = [{'sex_Male': 1, 'race_White': 0}, {'sex_Male': 0}]

    # Calcolo della metrica sul dataset originale
    overall_metric_original = BinaryLabelDatasetMetric(dataset=aif_overall_dataset, unprivileged_groups=overall_unprivileged_groups, privileged_groups=overall_privileged_groups)

    # stampiamo la metrica mean_difference sul file di report    
    print_fairness_metrics('mean_difference',overall_metric_original.mean_difference(),'Overall mean_difference before')
    
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    RW = Reweighing(unprivileged_groups=overall_unprivileged_groups,privileged_groups=overall_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    overall_dataset_transformed = RW.fit_transform(aif_overall_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    overall_metric_transformed = BinaryLabelDatasetMetric(dataset=overall_dataset_transformed,unprivileged_groups=overall_unprivileged_groups,privileged_groups=overall_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics('mean_difference',overall_metric_transformed.mean_difference(),'Overal mean_difference value after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics('mean_difference',overall_metric_original.num_positives(privileged=True),'(OVR) Num. of positive instances of priv_group before')
    print_fairness_metrics('mean_difference',overall_metric_original.num_positives(privileged=False),'(OVR) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics('mean_difference',overall_metric_transformed.num_positives(privileged=True),'(OVR) Num. of positive instances of priv_group after')
    print_fairness_metrics('mean_difference',overall_metric_transformed.num_positives(privileged=False),'(OVR) Num. of positive instances of unpriv_group after')
    #
    #
    # osservando il risultato di questa operazione scopriamo che il numero di individui bianchi di sesso maschile è più grande del resto degli individui
    # (maschi di razza diversa/donne), ma nonostante questo il dataset presenta una mean_difference negativa, quindi, in generale, il gruppo più piccolo,
    # indicato inizialmente come sfavorito date le informazioni precedenti, in realtà ha un numero più alto di predizioni positive.
    # Questo porta quindi ad un ribilanciamento che favorisce ulteriormente il gruppo più piccolo indicato come sfavorito, aumentando maggiormente
    # la mean_difference con il gruppo indicato come favorito.
    # Per questo possiamo concludere che questo ribilanciamento non è corretto
    #
    #
    # ritorniamo alla funzione chiamante l'unico dataset in cui è stato possibile evidenziare e rimuovere disparità fra i gruppi individuati per poter
    # addestrare un modello più fair
    aif_df = aif_race_dataset.convert_to_dataframe()[0]

    return aif_df

def print_fairness_metrics(metric_name, metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/adult_{metric_name}_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

load_dataset()