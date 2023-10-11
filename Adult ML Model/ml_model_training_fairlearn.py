import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import *
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # print di debug
    # print(df.head)

    training_model(df)

def training_model(dataset):
    ## funzione di sviluppo del modello

    # drop delle features superflue
    dataset.drop("ID",axis=1,inplace=True)

    # lista con tutte le features del dataset
    features = dataset.columns.tolist()

    # drop dalla lista del nome della variabile target
    features.remove('salary')

    # setting lista contenente nomi degli attributi protetti
    protected_features_names = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male']

    sex_features = ['sex_Female','sex_Male']
    race_features = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White']
    # settiamo delle metriche utili per poter fornire delle valutazioni sugli attributi sensibili tramite il framework FairLearn
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "selection rate": selection_rate,
        "count": count,
    }

    # setting del set contenente le features utili all'apprendimento
    X = dataset[features]

    # setting del set contenente la feature target
    y = dataset['salary']

    # setting del set contenente le features protette
    protected_features = dataset[protected_features_names]

    # setting del set contenente il sesso degli individui presenti nel dataset
    sex = dataset[sex_features]

    # setting del set contenente razza degli indivuidi presenti nel dataset
    race = dataset[race_features]

    # setting pipeline contenente modello e scaler per ottimizzazione dei dati da fornire al modello
    model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # setting pipeline da addestrare sul dataset soggetto ad operazioni di fairness
    fair_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

    # richiamiamo la funzione che dal dataset originale genera un nuovo dataset modificato rimuovendo la correlazione fra gli attributi sensibili e non
    # del dataset
    fair_dataset = fairness_preprocess_op(dataset=dataset,protected_features_names=protected_features_names)

    # estraiamo feature X ed y dal dataset ricalibrato
    X_fair = fair_dataset[features]
    y_fair = fair_dataset['salary']

    # setting della strategia KFold con standard di 10 gruppi
    kf = KFold(n_splits=10)

    # inizializzo contatore per il ciclo KFold
    i = 0

    # setting array contenente valori del dataframe
    df_array = np.asarray(dataset)

    # ciclo strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        # setting traing set X ed y dell'iterazione i-esima
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # setting training set delle sole varibili protette dell'iterazione i-esima
        protected_features_train = protected_features.iloc[train_index]

        # setting training set della singole variabili protette contenenti informazioni sul sesso e razza dell'individuo
        sex_train = sex.iloc[train_index]
        race_train = race.iloc[train_index]

        # setting test set X ed y dell'iterazione i-esima
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # setting training set delle sole varibili protette dell'iterazione i-esima
        protected_features_test = protected_features.iloc[test_index]

        # setting test set della singole variabili protette contenenti informazioni sul sesso e razza dell'individuo
        sex_test = sex.iloc[test_index]
        race_test = race.iloc[test_index]

        # training modello sul set X ed y dell'iterazione i-esima
        model_pipeline.fit(X_train,y_train)

        # produciamo una predizione di test per l'iterazione i-esima
        pred = model_pipeline.predict(X_test)

        # calcoliamo delle metriche di fairness sulla base degli attributi sensibili
        # overall_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=protected_features_test)
        # mf.by_group.plot.bar(
        #     subplots=True,
        #     layout=[3, 3],
        #     legend=False,
        #     figsize=[12, 8],
        #     title="Show all metrics",
        # )

        # calcoiamo delle metriche di fairness sulla base dell'attributo protetto "sex"
        sex_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=sex_test)
        sex_mf_no_zeros = sex_mf.by_group
        sex_mf_no_zeros = sex_mf_no_zeros.dropna()
        sex_mf_no_zeros.plot.bar(
            subplots=True,
            layout=[2, 2],
            legend=False,
            figsize=[20, 10],
            title="Show all metrics",
        )

        # calcoiamo delle metriche di fairness sulla base dell'attributo protetto "race"
        race_mf = MetricFrame(metrics=metrics,y_true=y_test,y_pred=pred,sensitive_features=race_test)
        race_mf_no_zeros = race_mf.by_group
        race_mf_no_zeros = race_mf_no_zeros.dropna()
        race_mf_no_zeros.plot.bar(
            subplots=True,
            layout=[3, 2],
            legend=False,
            figsize=[20, 10],
            title="Show all metrics",
        )

        validate(model_pipeline,'std_models', i, X_test, y_test)

        # addestriamo ora un modello sul dataframe in precedenza ricalibrato usando fairlearn
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        fair_model_pipeline.fit(X_fair_train,y_fair_train)

        validate(fair_model_pipeline, 'fair_models', i, X_fair_test, y_fair_test)
    
    # per stampare i grafici generati
    plt.show()

def fairness_preprocess_op(dataset, protected_features_names):
    ## funzione che utilizza classe offerta da fairlearn in grado di mitigare la correlazione fra gli attributi sensibili e non del dataset

    # lista contenente nomi colonne del dataset escluse le feature sensibili
    features_names = dataset.columns.tolist()
    for feature in protected_features_names:
        features_names.remove(feature)

    # creiamo un oggetto fornito dalla libreria Fairlean in grado di rimuovere correlazione fra le feature del dataset e le feature sensibili
    corr_remover = CorrelationRemover(sensitive_feature_ids=protected_features_names,alpha=1.0)
    fair_dataset = corr_remover.fit_transform(dataset)

    # ricostruiamo il dataframe inserendo le feature appena modificate
    fair_dataset = pd.DataFrame(
        fair_dataset, columns=features_names
    )

    # inseriamo nel nuovo dataframe le variabili sensibili rimosse in precedenza e la variabile target
    fair_dataset[protected_features_names] = dataset[protected_features_names]
    fair_dataset['salary'] = dataset['salary']

    # stampiamo heatmap che confrontano il grado di correlazione fra gli attributi sensibili e alcuni attributi del dataset
    show_correlation_heatmap(dataset=dataset,title='original dataset')
    show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset


def show_correlation_heatmap(dataset,title):
    ## funzione che genera heatmap sul dataset fornito usando attributi sensibili e non per mostrare il grado di correlazione presente

    # settiamo una lista contenente alcuni attrubi del dataset e le variabili sensibili
    sens_features_and_unses_features = ['sex_Male','sex_Female','race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','age','education-num','hours-per-week','salary','workclass_Federal-gov']
    
    # creiamo una heatmap del dataset che mostra la correlazione fra gli attributi dichiarati in precedenza
    sns.heatmap(dataset[sens_features_and_unses_features].corr(),annot=True,cmap='coolwarm',fmt='.2f',)
    plt.title(title)
    plt.show()

def validate(ml_model, model_type, index, X_test, y_test):
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
    with open(f"./reports/{model_type}/fairlearn/adult_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_type}/fairlearn/adult_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC report: {auc_score}')
        f.write('\n')



load_dataset()