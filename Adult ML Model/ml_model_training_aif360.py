import numpy as np 
import pandas as pd 
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from codecarbon import track_emissions
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
import xgboost as xgb
from sklearn.svm import SVC
import pickle
import warnings

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset e drop features superflue

    df = pd.read_csv('./Adult Dataset/adult_modificato.csv')

    # drop ID dal dataset
    df.drop('ID',inplace=True,axis=1)

    training_model(df)


def training_model(dataset):
    ## funzione di apprendimento del modello sul dataset

    # setting variabili protette
    protected_features_names = [
        'race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','sex_Female','sex_Male'
    ]

    fair_dataset = dataset.copy(deep=True)

    # creiamo un dataset fair effettuando modifiche al dataset originale
    sample_weights = test_fairness(dataset)

    fair_dataset['weights'] = sample_weights

    # setting nomi features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo il nome della feature target dalla lista nomi features
    features.remove('salary')

    # setting nome target feature
    target = ['salary']

    # setting dataset features
    X = dataset[features]
    X_fair = fair_dataset[features]

    # setting dataset target feature
    y = dataset[target]
    y_fair = fair_dataset[target]

    sample_weights = fair_dataset['weights']

    # setting strategia KFold con 10 iterazioni standard
    kf = KFold(n_splits=10)

    # trasformiamo dataset in array per estrarre indici per la strategia KFold
    df_array = np.asarray(dataset)

    # setting contatore iterazioni KFold
    i = 0

    # costruiamo il modello standard tramite pipeline contenente uno scaler per la normalizzazione dati e un regressore
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression(max_iter=200))
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    # costruiamo un modello tramite pipeline su cui utilizzare un dataset opportunamente modificato per aumentare fairness
    lr_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',LogisticRegression())
    ])

    rf_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',RandomForestClassifier())
    ])

    svm_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',SVC(probability=True))
    ])

    xgb_fair_model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',xgb.XGBClassifier(objective='binary:logistic',random_state=42))
    ])

    # ciclo strategia KFold per il modello base
    for train_index,test_index in kf.split(df_array):
        i = i+1

        # setting training set per l'i-iterazione della strategia KFold 
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # setting test set per l'i-iterazione della sstrategia KFold 
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # training del modello base sul training set dell'i-esima iterazione
        lr_model_pipeline.fit(X_train,y_train.values.ravel())
        rf_model_pipeline.fit(X_train,y_train.values.ravel())
        svm_model_pipeline.fit(X_train,y_train.values.ravel())
        xgb_model_pipeline.fit(X_train,y_train.values.ravel())

        # calcolo metriche di valutazione sul modello base dell'i-esima iterazione 
        validate(lr_model_pipeline,i,"std_models",'lr',X_test,y_test)
        validate(rf_model_pipeline,i,'std_models','rf',X_test,y_test)
        validate(svm_model_pipeline,i,'std_models','svm',X_test,y_test)
        validate(xgb_model_pipeline,i,'std_models','xgb',X_test,y_test)
        
        # setting training set per l'i-esima iterazione della strategia KFold per il modello fair
        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]
        sample_weights_train = sample_weights[train_index]

        # setting test set per l'i-esima iterazione della strategia KFold per il modello fair
        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # training del modello sul training set dell'i-esima iterazione
        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)
        xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(), model__sample_weight=sample_weights_train)

        # calcolo metriche di valutazione sul modello fair dell'i-esima iterazione
        validate(lr_fair_model_pipeline,i,'fair_models','lr',X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,i,'fair_models','rf',X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,i,'fair_models','svm',X_fair_test,y_fair_test)
        validate(xgb_fair_model_pipeline,i,'fair_models','xgb',X_fair_test,y_fair_test)

        processed_train = processing_fairness(dataset,X_train,y_train,protected_features_names,i)

        X_postop_train = processed_train[features]
        y_postop_train = processed_train['salary']
    
        post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
        post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
        post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
        post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

        validate_postop(post_lr_model_pipeline,'lr',i,X_test,y_test)
        validate_postop(post_rf_model_pipeline,'rf',i,X_test,y_test)
        validate_postop(post_svm_model_pipeline,'svm',i,X_test,y_test)
        validate_postop(post_xgb_model_pipeline,'xgb',i,X_test,y_test)

    with open('./reports/final_scores/aif360/adult_scores.txt','w') as f:
        f.write(str(lr_model_pipeline.score(X,y)))
        f.write(str(rf_model_pipeline.score(X,y)))
        f.write(str(svm_model_pipeline.score(X,y)))
        f.write(str(xgb_model_pipeline.score(X,y)))
        
        f.write(str(lr_fair_model_pipeline.score(X_fair,y_fair)))
        f.write(str(rf_fair_model_pipeline.score(X_fair,y_fair)))
        f.write(str(svm_model_pipeline.score(X_fair,y_fair)))
        f.write(str(xgb_fair_model_pipeline.score(X_fair,y_fair)))

        f.write(str(post_lr_model_pipeline.score(X,y)))
        f.write(str(post_rf_model_pipeline.score(X,y)))
        f.write(str(post_svm_model_pipeline.score(X,y)))
        f.write(str(post_xgb_model_pipeline.score(X,y)))

    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_aif360_adult_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_aif360_adult_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_aif360_adult_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_aif360_adult_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_aif360_adult_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_aif360_adult_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_aif360_adult_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_aif360_adult_model.sav','wb'))
    pickle.dump(post_lr_model_pipeline,open('./output_models/postop_models/lr_aif360_adult_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/postop_models/rf_aif360_adult_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/postop_models/svm_aif360_adult_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/postop_models/xgb_aif360_adult_model.sav','wb'))

def processing_fairness(dataset,X_set,y_set,protected_features,index):

    fair_classifier = MetaFairClassifier(type='fdr')

    train_dataset = pd.DataFrame(X_set)

    train_dataset['salary'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'race_White': 1}]
    unprivileged_groups = [{'race_White': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    if index == 1:
        first_message = True
    else:
        first_message = False
    
    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Race Mean difference pre inprocessing',first_message=first_message)
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Race DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Race Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Race DI post inprocessing')

    privileged_groups = [{'sex_Male': 1}]
    unprivileged_groups = [{'sex_Female': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Gender Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Gender DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Gender Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Gender DI post inprocessing')

    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train
    

def print_postop_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/aif360/adult_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

def validate_postop(ml_model,model_type,index,X_test,y_test):
    pred = ml_model.predict(X_test)

    report = classification_report(y_pred=pred,y_true=y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if index == 1:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/postop_models/aif360/adult/{model_type}_adult_metrics_report.txt',open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write('Metriche di valutazione:')
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')


def validate(ml_model,index,model_vers,model_type,X_test,y_test):
    ## funzione utile a calcolare le metriche di valutazione del modello passato in input

    pred = ml_model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)
    
    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    report = classification_report(y_test, pred)

    if index == 1:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file matrice di confusione ottenuta
    with open(f"./reports/{model_vers}/aif360/adult/{model_type}_adult_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/aif360/adult/{model_type}_adult_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def test_fairness(original_dataset):
    ## funzione che testa la fairness del dataset tramite libreria AIF360 e restituisce un dataset fair opportunamente modificato

    race_features = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "race"
    aif_race_dataset = BinaryLabelDataset(
        df=original_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_features,
        privileged_protected_attributes=['race_White']
    )
    
    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui "bianchi" come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di razze diverse, che vengono settati come appartenenti al gruppo sfavorito.
    race_privileged_groups = [{'race_White': 1}]
    race_unprivileged_groups = [{'race_White': 0}]

    # Calcolo della metrica sul dataset originale
    race_metric_original = BinaryLabelDatasetMetric(dataset=aif_race_dataset, unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)    
    
    # stampiamo la metrica mean_difference sul file di report    
    # (differenza fra predizioni positive di indivudi sfavoriti rispetto alle predizioni positive degli individui favoriti)
    print_fairness_metrics(race_metric_original.mean_difference(),'Race mean_difference before',first_message=True)
    print_fairness_metrics(race_metric_original.disparate_impact(),'Race DI before')
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    race_dataset_transformed = RACE_RW.fit_transform(aif_race_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_dataset_transformed,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(race_metric_transformed.mean_difference(),'Race mean_difference after')
    print_fairness_metrics(race_metric_transformed.disparate_impact(),'Race DI after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(race_metric_original.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group before')
    print_fairness_metrics(race_metric_original.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(race_metric_transformed.num_positives(privileged=True),'(RACE) Num. of positive instances of priv_group after')
    print_fairness_metrics(race_metric_transformed.num_positives(privileged=False),'(RACE) Num. of positive instances of unpriv_group after')
    
    # passiamo ora a valutare il dataset sulla base della feature legata al genere

    new_dataset = race_dataset_transformed.convert_to_dataframe()[0]

    sample_weights = race_dataset_transformed.instance_weights

    new_dataset['weights'] = sample_weights

    # setting nome varibili sensibili legate al sesso
    sex_features = ['sex_Male','sex_Female']

    # costruiamo il dataset sfruttando l'oggetto richiesto dalla libreria AIF360 per operare
    # questo dataset sfrutterà solamente i gruppi ottenuti utilizzando la feature "sex"
    aif_sex_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
        instance_weights_name=['weights']
    )

    # setting dei gruppi privilegiati e non del delle varibili protette
    # in particolare, scegliamo di trattare gli individui di sesso maschile come favoriti data la forte presenza di quest'ultimi all'interno del dataset
    # rispetto agli individui di sesso femminile, che vengono settati come appartenenti al gruppo sfavorito.
    sex_privileged_groups = [{'sex_Male': 1}]
    sex_unprivileged_groups = [{'sex_Female': 1}]

    # Calcolo della metrica sul dataset originale
    sex_metric_original = BinaryLabelDatasetMetric(dataset=aif_sex_dataset, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups) 
    
    # stampiamo la metrica mean_difference sul file di report    
    print_fairness_metrics(sex_metric_original.mean_difference(),'Sex mean_difference before')
    print_fairness_metrics(sex_metric_original.disparate_impact(),'Sex DI before')
    
    # creiamo l'oggetto reweighing offerto dalla lib AIF360 che permette di bilanciare le istanze del dataset fra i gruppi indicati come favoriti e sfavoriti
    SEX_RW = Reweighing(unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    # bilanciamo il dataset originale sfruttando l'oggetto appena creato
    sex_dataset_transformed = SEX_RW.fit_transform(aif_sex_dataset)

    # vengono ricalcolate le metriche sul nuovo modello appena bilanciato
    sex_metric_transformed = BinaryLabelDatasetMetric(dataset=sex_dataset_transformed,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)
    
    # stampa della mean_difference del nuovo modello bilanciato sul file di report
    print_fairness_metrics(sex_metric_transformed.mean_difference(),'Sex mean_difference value after')
    print_fairness_metrics(sex_metric_transformed.disparate_impact(),'Sex DI value after')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi favoriti e sfavoriti prima del bilanciamento
    print_fairness_metrics(sex_metric_original.num_positives(privileged=True),'(SEX) Num. of positive instances of priv_group before')
    print_fairness_metrics(sex_metric_original.num_positives(privileged=False),'(SEX) Num. of positive instances of unpriv_group before')

    # vengono stampate sul file di report della metrica anche il numero di istanze positive per i gruppi post bilanciamento
    print_fairness_metrics(sex_metric_transformed.num_positives(privileged=True),'(SEX) Num. of positive instances of priv_group after')
    print_fairness_metrics(sex_metric_transformed.num_positives(privileged=False),'(SEX) Num. of positive instances of unpriv_group after')

    sample_weights = sex_dataset_transformed.instance_weights

    return sample_weights

def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/adult_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dataset()