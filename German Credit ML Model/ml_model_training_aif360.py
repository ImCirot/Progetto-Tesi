import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from codecarbon import track_emissions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


@track_emissions(offline=True, country_iso_code="ITA")
def traning_and_testing_model():
    ## Funzione per il training e testing del modello scelto

    df = pd.read_csv("./German Credit Dataset/dataset_modificato.csv")

    # Drop delle colonne superflue
    df.drop('ID', inplace=True ,axis=1)

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    df['Target'] = df['Target'].replace(2,0)

    # print di debug
    # pd.options.display.max_columns = 2
    # print(df.head())

    (fair_dataset,sample_weights) = test_fairness(df)
    fair_dataset['weights'] = sample_weights

    features = df.columns.tolist()
    features.remove('Target')

    target = ['Target']

    X = df[features]
    X_fair = fair_dataset[features]

    y = df[target]
    y_fair = df[target]

    sample_weights = fair_dataset['weights']

    # Si crea un array del dataframe utile per la KFold
    df_array = np.array(df)

    # Settiamo il numero di gruppi della strategia KFold a 10
    kf = KFold(n_splits=10)

    # inizializiamo contatore i
    i = 0

    # Creiamo due pipeline che effettuano delle ulteriori operazioni di scaling dei dati per addestriare il modello
    # in particolare la pipeline standard sarà addestrata sui dati as-is
    # mentre la fair pipeline verrà addestrata su dati sui vengono applicate strategie di fairness
    # volte a rimuovere discriminazione e bias nel dataset di training
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))

    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',LogisticRegression(class_weight={1:1,0:5}))
    ])
    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',RandomForestClassifier(class_weight={1:1,0:5}))
    ])
    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler',StandardScaler()), 
        ('model',SVC(probability=True,class_weight={1:1,0:5}))
    ])

    # Strategia KFold
    for train_index, test_index in kf.split(df_array):
        i = i+1

        # setting del training set dell'i-esima iterazione 
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]

        # setting del test set dell'i-esima iterazione 
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]

        # fit del modello sul training set dell'i-esima iterazione
        lr_model_pipeline.fit(X_train,y_train.values.ravel())
        rf_model_pipeline.fit(X_train,y_train.values.ravel())
        svm_model_pipeline.fit(X_train,y_train.values.ravel())

        # Stampiamo metriche di valutazione per il modello
        validate(lr_model_pipeline, i, "std_models", 'lr', X_test, y_test)
        validate(rf_model_pipeline,i,'std_models','rf',X_test,y_test)
        validate(svm_model_pipeline,i,'std_models','svm',X_test,y_test)
    
    # costruiamo array dal dataset ricalibrato per attuare strategia KFold
    fair_array = np.asarray(fair_dataset)

    # reset contatore i
    i = 0

    for train_index,test_index in kf.split(fair_array):
        i = i+1

        # setting training set dell'i-esima iterazione dal dataset ricalibrato
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]
        sample_weights_train = sample_weights[train_index]

        # setting test set dell'i-esima iterazione dal dataset ricalibrato
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # fit del modello sul training set dell'i-esima iterazione
        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=sample_weights_train)

        # Stampiamo metriche di valutazione per il modello
        validate(lr_fair_model_pipeline, i, "fair_models", 'lr', X_fair_test, y_fair_test)
        validate(rf_fair_model_pipeline,i,'fair_models','rf',X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,i,'fair_models','svm',X_fair_test,y_fair_test)

    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_aif360_credit_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_aif360_credit_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_aif360_credit_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_aif360_credit_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_aif360_credit_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_aif360_credit_model.sav','wb'))
            
def validate(ml_model,index,model_vers,model_type,X_test,y_test):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

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
    with open(f"./reports/{model_vers}/aif360/credit/{model_type}_credit_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/aif360/credit/{model_type}_credit_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def test_fairness(dataset):
    ## Funzione che presenta alcune metriche di fairness sul dataset utilizzato e applica processi per ridurre/azzerrare il bias

    # Attributi sensibili
    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    # Attributi sensibili con vantaggio
    privileged_attribute_names = [
        'sex_A91','sex_A92','sex_A94'
    ]

    aif360_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_attribute_names,
    )

    # Setting dei gruppi privilegiati e non
    # In questo caso si è scelto di trattare come gruppo privilegiato tutte le entrate che presentano la feature 'Sex_A94' = 1, ovvero tutte le entrate
    # che rappresentano un cliente maschio sposato/vedovo. Come gruppi non privilegiati si è scelto di utilizzare la feature 'sex_94' != 1,
    # ovvero tutti gli altri individui.
    privileged_groups = [{'sex_A93': 1}]
    unprivileged_groups = [{'sex_A93': 0}]

    # Calcolo della metrica sul dataset originale
    metric_original = BinaryLabelDatasetMetric(dataset=aif360_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)    
    
    # Se la metrica originale ottiene già valore 0.0, allora il dataset è gia equo e non ha bisogno di ulteriori operazioni
    if(metric_original.mean_difference() != 0.0):
        # Utilizzamo un operatore di bilanciamento offerto dall'API AIF360
        RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

        # Bilanciamo il dataset
        dataset_transformed = RW.fit_transform(aif360_dataset)
        # Ricalcoliamo la metrica
        metric_transformed = BinaryLabelDatasetMetric(dataset=dataset_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # stampa della mean_difference del modello originale
    print_fairness_metrics(metric_original.mean_difference(),'Mean_difference value before', first_message=True)
    print_fairness_metrics(metric_original.disparate_impact(),'DI value before')

    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(metric_transformed.mean_difference(),'Mean_difference value after')
    print_fairness_metrics(metric_transformed.disparate_impact(),'DI value after')
    
    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(metric_original.num_positives(privileged=True),'Num. of positive instances of priv_group before')
    print_fairness_metrics(metric_original.num_positives(privileged=False),'Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(metric_transformed.num_positives(privileged=True),'Num. of positive instances of priv_group after')
    print_fairness_metrics(metric_transformed.num_positives(privileged=False),'Num. of positive instances of unpriv_group after')

    # Creiamo un nuovo dataframe sulla base del modello ripesato dall'operazione precedente
    fair_dataset = dataset_transformed.convert_to_dataframe()[0]

    return (fair_dataset,dataset_transformed.instance_weights)
    
def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/credit_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

# Chiamata funzione inizale di training e testing
traning_and_testing_model()