import ucimlrepo
import pandas as pd

def load_dataset():
    ## funzione di fetch del dataset dalla repository UCI

    # fetch del dataset dalla repo UCI
    dataset = ucimlrepo.fetch_ucirepo(id=697)

    # estraiamo il dataframe dall'oggetto fetchato
    df = dataset.data.original

    # rimpiazziamo le classi categoriche della variabile Target con valori numerici
    # il problema si presenta come problema di classificazione per 3 categorie,
    # per poter studiare la fairness delle valutazioni ho ridotto a 2 le categorie,
    # trattando il possibile outcome 'enrolled' come 0, in quanto rappresenta uno studente
    # iscritto al corso ma che non ha completato ancora il percorso al termine del periodo canonico. 
    # Questa scelta di considerare questo outcome come 'negativo' è stata presa dalla caratteristica
    # stessa della condizione, ovvero uno studente che non ha scelto di abbandonare gli studi
    # ma non ha comunque ancora concluso nel termine canonico indicato.
    df['Target'] = df['Target'].replace('Dropout',0)
    df['Target'] = df['Target'].replace('Enrolled',0)
    df['Target'] = df['Target'].replace('Graduate',1)

    # salviamo il dataframe ottenuto in un file csv da utilizzare per i modelli
    df.to_csv('./Student Dataset/dataset.csv',index_label='ID')

load_dataset()