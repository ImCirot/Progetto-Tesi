import ucimlrepo as uci
import pandas as pd
import numpy as np

def load_dataset():
    dataset = uci.fetch_ucirepo(id=45)

    df = dataset.data.original

    print(df)

    df = df.dropna()

    df['age'] = df['age'].apply(lambda x: 1 if x>=40 else 0)
    
    print(df.head)
    # poichè non è importante ai fini dello studio classificare correttamente lo stato della malattia, consideriamo come 1 tutti gli stati di malattia presenti
    # mentre 0 se la malattia è completamentamente assente, in questo modo i modelli saranno in grado di prevedere la totale assenza di malattia o la presenza 
    # di quest'ultima
    df['num'] = df['num'].replace({2:1,3:1,4:1})

    df.to_csv('./Heart Disease Dataset/dataset.csv',index_label='ID')

load_dataset()