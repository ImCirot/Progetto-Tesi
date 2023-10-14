import ucimlrepo
import pandas as pd

def load_dataset():
    ## funzione di fetch del dataset dalla repository UCI

    # fetch del dataset dalla repo UCI
    dataset = ucimlrepo.fetch_ucirepo(id=697)

    # estraiamo il dataframe dall'oggetto fetchato
    df = dataset.data.original

    # rimpiazziamo le classi categoriche della variabile Target con valori numerici
    df['Target'] = df['Target'].replace('Dropout',0)
    df['Target'] = df['Target'].replace('Graduate',1)

    # salviamo il dataframe ottenuto in un file csv da utilizzare per i modelli
    df.to_csv('./Student Dataset/dataset.csv',index_label='ID')

load_dataset()