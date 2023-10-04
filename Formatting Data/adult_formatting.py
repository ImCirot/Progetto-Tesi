import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def formatting_data():

    df = pd.read_csv('./Adult Dataset/adult_dataset.csv', na_values=' ?')

    # cancelliamo dal dataset le entry con attributi mancanti
    df = df.dropna()

    # lista di feature numeriche che non necessitano di modifiche
    numeric_features = [
        'age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'
    ]

    # lista delle feature categoriche
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country' 
    ]

    # lista nomi delle feature del dataset
    features = df.columns.to_list()

    # ciclo per rimuovere le feature numeriche dalla lista delle feature
    for num_feature in numeric_features:
        features.remove(num_feature)

    # ciclo per rimuovere gli spazi superflui presenti nei valori delle stringhe
    for feature in features:
        df[feature] = df[feature].apply(lambda x: x.strip())

    # rimpiazziamo i valori categorici della nostra variabile target con valori numerici
    # 1 per salario maggiore di 50K, 0 altrimenti
    df['salary'] = df['salary'].replace('<=50K',0)
    df['salary'] = df['salary'].replace('>50K',1)

    # encoding delle feature categoriche in valori numerici tramite la funzione get_dummies()
    # funzione che crea X nuove colonne, con X numero di possibili valori per la feature, impostando valore 0.0 e 1.0
    one_hot = pd.get_dummies(df[categorical_features], dtype=int)

    # Cancelliamo dal DataFrame originale le features categoriche espanse per poi unire alle rimanenti il nuovo dataframe ottenuto dalla funzione
    # get_dummies()
    df = df.drop(categorical_features, axis=1)
    df = df.join(one_hot)

    # stampiamo in output il dataset in un nuovo file .csv pronto per essere utilizzato
    df.to_csv('./Adult Dataset/adult_modificato.csv',index_label='ID')

formatting_data()
