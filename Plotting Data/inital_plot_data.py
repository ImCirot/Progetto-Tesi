import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import aif360 as aif
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
ohe_encoder = OneHotEncoder(sparse_output=False)

g = np.asarray(g).reshape(-1,1)

g = ohe_encoder.fit_transform(g)

# Assegniamo i valori ottenuti dall'encoding al dataset originale
dataset[features] = X
dataset['Target'] = y

g_array = np.asarray(g)
i = 0

dataset.rename(columns={"Personal status and sex":"Sex_0"}, inplace=True)
dataset['Sex_0'] = g[:, 0]
while i < len(g_array):
    i = i + 1
    dataset.insert(loc=(8+i), column=f"Sex_{i}", value=g[:, i])

check = ['Sex_0','Sex_1']
print(dataset[check])