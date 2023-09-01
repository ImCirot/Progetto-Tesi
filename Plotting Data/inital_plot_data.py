import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import aif360 as aif
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from codecarbon import track_emissions

@track_emissions(offline=True, country_iso_code="ITA")
def loading_dataset():
    ## funzione che carica il dataset in un dataframe, effettua un encoding delle feature tramite OneShot per le varibili categoriche 
    ## e salva il nuovo dataframe pronto per essere utilizzato dal modello
    
    # Creazione etichette del dataset
    col_names = [
        'Status of exisiting checking account',
        'Duration in month',
        'Credit history',
        'Purpose',
        'Credit amount',
        'Savings account/bonds',
        'Present employment since',
        'Installment rate in percentage of disposable income',
        'sex',
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
        'Target'
    ]
    
    # Etichette del dataset contenenti dati di tipo categorico
    # Questa lista viene utilizzata per indicare quali tipi di dati andranno codificati tramite OneShot Encoding
    categorical_features = [
        'Status of exisiting checking account',
        'Credit history',
        'Purpose',
        'Savings account/bonds',
        'Present employment since',
        'sex',
        'Other debtors / guarantors',
        'Property',
        'Other installment plans',
        'Housing',
        'Job',
        'Telephone',
        'foreign worker'
    ]

    # Feature numeriche del dataset che non richiedono alcuna conversione
    numerical_features = [
        'Duration in month',
        'Credit amount',
        'Installment rate in percentage of disposable income',
        'Present residence since',
        'Age in years',
        'Number of existing credits at this bank',
        'Number of people being liable to provide maintenance for',
        'Target'
    ]

    # Lettura del dataset da file memorizzato nel DataFrame df
    df = pd.read_csv('./Dataset/German-Dataset.csv', names=col_names, index_col=False, header=None)

    # Tramite la funzione 'get_dummies()' possiamo indicare features del dataset di tipo categorico da trasormare in numerico, la funzione
    # ci restituisce un dataframe contenente le feature indicate codificate in maniera numerica espandendo il numero di colonne per ogni possibile valore
    # Es. una variabile categorica che può assumere 4 diversi valori verrà espansa in 4 colonne ognuna con valore 0/1 in base al valore categorico 
    # corrispondente.
    one_hot = pd.get_dummies(df[categorical_features], dtype=int)

    # Cancelliamo dal DataFrame originale le features categoriche espanse per poi unire alle rimanenti il nuovo dataframe ottenuto dalla funzione
    # get_dummies()
    df = df.drop(categorical_features, axis=1)
    df = df.join(one_hot)

    # Stampa di debug
    print(df.head)

    # Salviamo in locale il dataset in un file csv, pronto per essere utilizzato per la fase di training e testing del modello
    ouptut = df.to_csv('./Dataset/dataset_modificato.csv', index_label="ID")

# Chiamata funzione per caricare il dataset e organizzare le features per poter essere utilizzate in fase di training
loading_dataset()
