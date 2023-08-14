import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import aif360 as aif
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from codecarbon import track_emissions

@track_emissions
def loading_dataset():
    ## funzione che carica il dataset in un dataframe, effettua un encoding delle feature tramite OneShot per le varibili sensibili e Ordinal per le rimanenti feature
    
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
        'Personal status and sex',
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

    # Creazione di un dataframe del dataset fornito di etichette
    dataset = pd.read_csv('./Dataset/German-Dataset.csv', index_col=False, header=None, names=col_names)

    data = dataset.values

    features = ['Status of exisiting checking account',
        'Duration in month',
        'Credit history',
        'Purpose',
        'Credit amount',
        'Savings account/bonds',
        'Present employment since',
        'Installment rate in percentage of disposable income',
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

    X = dataset[features]
    y = dataset['Target']
    g = dataset['Personal status and sex']

    # OrdinalEncoding delle variabili categoriche e della nostra variabile da predire
    encoder = OrdinalEncoder()
    label_encoder = LabelEncoder()
    encoder.fit(X)
    label_encoder.fit(y)
    X = encoder.transform(X)
    y = label_encoder.transform(y)

    # OneShotEncoding della variabile protetta
    # la nuova varibile g conterrà quattro nuove colonne, una per ogni possibile valore della varibile originale.
    # È da notare come, nella descrizione del dataset, i valori possibili siano 5 e l'output che otteniamo dell'encoding sia 4.
    # Questo avviene poichè il valore A95, ovvero donna single, non appare mai nelle 1000 entrate del dataset e dunque non viene considerato.
    ohe_encoder = OneHotEncoder(sparse_output=False)

    g = np.asarray(g).reshape(-1,1)

    g = ohe_encoder.fit_transform(g)

    # Assegniamo i valori ottenuti dall'encoding al dataset originale
    dataset[features] = X
    dataset['Target'] = y

    g_array = np.asarray(g)
    i = 1

    # Espandiamo la feature "Personal status and sex" in 4 nuove feature che rappresentano i possibili stati della varibile originale
    # in questo modo otteniamo:
    # Sex_0: A93 -> Uomo single
    # Sex_1: A92 -> Donna divorziata/separata/sposata
    # Sex_2: A91 -> Uomo divorziato/separato
    # Sex_3: A94 -> Uomo sposato/vedovo
    # Come definito prima vengono create solamente 3 nuove feature poichè l'originale viene riutilizzata per l'attributo Sex_0,
    # e non essendo presente alcuna entrata con valore A95 è ridondante creare una feature apposita per quest'ultimo.
    dataset.rename(columns={"Personal status and sex":"Sex_0"}, inplace=True)
    dataset['Sex_0'] = g[:, 0]
    while i < 4:
        dataset.insert(loc=(8+i), column=f"Sex_{i}", value=0)
        dataset[f'Sex_{i}'] = g[:,i]
        i = i + 1

    check = ['Sex_0','Sex_1','Sex_2','Sex_3']
    print(dataset[check])

# Chiamata funzione per caricare il dataset e organizzare le features per poter essere utilizzate in fase di training
loading_dataset()