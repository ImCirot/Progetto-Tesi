import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.utils import resample


def load_dataset():
    df_train = pd.read_csv('./Home Credit Dataset/application_train.csv')

    df_train.drop('SK_ID_CURR',inplace=True,axis=1)
    
    df_train.drop('EMERGENCYSTATE_MODE',inplace=True,axis=1)

    categorical_features = [
        'OCCUPATION_TYPE',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE',

    ]

    num_features = df_train.columns.tolist()

    for feature in categorical_features:
        num_features.remove(feature)

    df_train[num_features] = df_train[num_features].fillna(value=-1)
    df_train[categorical_features] = df_train[categorical_features].fillna(value='Unknown')

    df_train_age= abs(df_train['DAYS_BIRTH']) // 365
    df_train_age = df_train_age.apply(lambda x: 1 if x>=25 else 0).astype(int)

    df_train['AGE_CAT'] = df_train_age

    df_train = df_train.drop('DAYS_BIRTH',axis=1)

    df_train['CODE_GENDER'] = df_train['CODE_GENDER'].map({'M':1,'F':0,'XNA':0})
    df_train['CODE_GENDER'] = df_train['CODE_GENDER'].astype(int)

    colonne_oggetto = df_train.select_dtypes(include=['object']).columns.tolist()

    df_dummies = pd.get_dummies(df_train[colonne_oggetto],dtype=int)

    df_train.drop(colonne_oggetto,axis=1,inplace=True)

    df_train = df_train.join(df_dummies)

    classe_maggioritaria = df_train[df_train['TARGET'] == 0]
    classe_minoritaria = df_train[df_train['TARGET'] == 1]

    classe_minoritaria_oversampled = resample(classe_minoritaria, replace=True, n_samples=len(classe_minoritaria)*2, random_state=42)

    classe_maggioritaria_undersampled = resample(classe_maggioritaria, replace=False, n_samples=len(classe_minoritaria_oversampled), random_state=42)

    df_bilanciato = pd.concat([classe_minoritaria_oversampled, classe_maggioritaria_undersampled])

    df_bilanciato = df_bilanciato.sample(frac=1, random_state=42).reset_index(drop=True)

    print(df_bilanciato['TARGET'].value_counts())
    print(df_bilanciato.shape)
    constant_columns = df_bilanciato.columns[df_bilanciato.nunique() == 1]

    df_bilanciato = df_bilanciato.drop(constant_columns,axis=1)

    df_bilanciato.to_csv('./Home Credit Dataset/dataset.csv',index_label='ID')

load_dataset()