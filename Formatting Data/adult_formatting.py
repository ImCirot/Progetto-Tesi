import numpy as np
import pandas as pd


def formatting_data():

    df = pd.read_csv('./Adult Dataset/adult_dataset.csv')
    
    # lista di feature numeriche che non necessitano di modifiche
    numeric_features = [
        'age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'
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

    df.to_csv('./Adult Dataset/adult_modificato.csv',index_label='ID')

formatting_data()
