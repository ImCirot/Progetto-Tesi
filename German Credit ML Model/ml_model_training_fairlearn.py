import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns

def training_model(dataset):
    ## funzione che addestra il modello sul dataset utilizzando strategia KFold

    # trasformiamo dataset in array per usare indici strategia KFold
    df_array = np.asarray(dataset)

    # evidenziamo le features utili alla predizione
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista features la feature target
    features.remove('Target')

    # evidenziamo gli attributi sensibili del dataset
    sex_features = [
        'sex_A91','sex_A92','sex_A93','sex_A94'
    ]

    # settiamo delle metriche utili per poter fornire delle valutazioni sugli attributi sensibili tramite il framework FairLearn
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "selection rate": selection_rate,
        "count": count,
    }

    # settiamo la nostra X sulle sole variabili di features
    X = dataset[features]

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    dataset['Target'] = dataset['Target'].replace(2,0)

    # settiamo la nostra y sulla variabile da predire
    y = dataset['Target']

    # settiamo un dataframe contenente solamente i valori degli attributi sensibili (utile per utilizzare il framework FairLearn)
    sex = dataset[sex_features]

    # settiamo contatore per ciclo KFold
    i = 0

    # settiamo il numero di ripetizioni uguale a 10, standard per la strategia KFold
    kf = KFold(n_splits=10)

    # Creiamo una pipeline contenente il modello basato su regressione logistica e uno scaler per poter scalare i dati correttamente per poter
    # utilizzare correttamente il modello
    model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))


    # proviamo a rimuovere eventuali correlazioni esistenti fra i dati e le features sensibli
    # utilizzando una classe messa a disposizione dalla libreria FairLearn
    corr_remover = CorrelationRemover(sensitive_feature_ids=sex_features,alpha=1.0)

    # creiamo un nuovo modello da addestrare sul dataset modificato
    fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression(class_weight={1:1,0:5}))

    features_names = dataset.columns.tolist()
    for sex_feature in sex_features:
        features_names.remove(sex_feature)
        

    # modifichiamo il set training usando lo strumento di preprocessing fornito dalla libreria FairLearn
    fair_dataset = corr_remover.fit_transform(dataset)
    fair_dataset = pd.DataFrame(
        fair_dataset, columns=features_names
    )

    fair_dataset[sex_features] = dataset[sex_features]
    fair_dataset['Target'] = dataset['Target']
    
    features_list = fair_dataset.columns.tolist()
    features_list.remove('Target')

    X_fair = fair_dataset[features_list]
    y_fair = fair_dataset['Target']
    
    sens_and_target_features = ['sex_A91','sex_A92','sex_A93','sex_A94','Duration in month','Credit amount','Installment rate in percentage of disposable income',
    'Present residence since','Age in years']
    sns.heatmap(dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    plt.title("Standard dataset heatmap")
    plt.show()

    sns.heatmap(fair_dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    plt.title("Modified dataset heatmap")
    plt.show()


    for train_index, test_index in kf.split(df_array):
        i = i + 1

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        sex_train = sex.loc[train_index]

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        sex_test = sex.loc[test_index]

        # fitting del modello sui dati di training per l'iterazione i-esima
        model_pipeline.fit(X_train,y_train)

        # produciamo una predizione di test per l'iterazione i-esima
        pred = model_pipeline.predict(X_test)

        # calcoliamo delle metriche di fairness sulla base degli attributi sensibili
        mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=sex_test)
        mf_no_zeros = mf.by_group
        mf_no_zeros = mf_no_zeros.dropna()
        mf_no_zeros.plot.bar(
            subplots=True,
            layout=[2, 2],
            legend=False,
            figsize=[20, 10],
            title="Show all metrics",
        )

        # validiamo i risultati prodotti dal modello all'iterazione i-esima chiamando una funzione che realizza metriche di valutazione
        validate(model_pipeline, 'std_models',i, X_test, y_test)

        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # addestriamo il modello
        fair_model_pipeline.fit(X_fair_train,y_train)

        # validiamo i risultati prodotti dal modello all'iterazione i-esima chiamando una funzione che realizza metriche di valutazione
        validate(fair_model_pipeline, 'fair_models',i, X_fair_test, y_fair_test)

    #     # proviamo alcune operazioni di postprocessing sul modello prodotto
    #     # postprocess_model = ThresholdOptimizer(
    #     #     estimator=model_pipeline,
    #     #     constraints='equalized_odds',
    #     #     objective='balanced_accuracy_score',
    #     #     prefit=True,
    #     #     predict_method='predict_proba'
    #     # )

    #     # postprocess_model.fit(X_train,y_train,sensitive_features=sex_train)
    #     # fair_pred = postprocess_model.predict(X_test,sensitive_features=sex_test)
    #     # fair_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=fair_pred,sensitive_features=sex_test)
    #     # mf.by_group.plot.bar(
    #     #     subplots=True,
    #     #     layout=[3, 3],
    #     #     legend=False,
    #     #     figsize=[12, 8],
    #     #     title="Show all metrics",
    #     # )
    # # per mostrare grafici
    # plt.show()

def validate(ml_model,model_type,index,X_test,y_test):
    ## funzione utile a calcolare metriche del modello realizzato

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    report = classification_report(y_test, pred)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/{model_type}/fairlearn/credit_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_type}/fairlearn/credit_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./German Credit Dataset/dataset_modificato.csv')

    df.drop('ID', axis=1, inplace=True)

    training_model(df)


load_dataset()