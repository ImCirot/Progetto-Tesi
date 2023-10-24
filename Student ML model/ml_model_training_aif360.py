from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
import xgboost as xgb
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import *
from fairlearn.reductions import DemographicParity
import pickle

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Student Dataset/dataset.csv')

    # drop ID dal dataframe
    df.drop('ID', inplace=True,axis=1)

    # richiamo funzione di training e testing dei modelli
    training_testing_models(df)

def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'Gender','Educational special needs',
        'Age at enrollment','Admission grade', 'International',
    ]

    sample_weights = test_fairness(dataset)

    fair_dataset = dataset.copy(deep=True)

    fair_dataset['weights'] = sample_weights

    feature_names = dataset.columns.tolist()

    feature_names.remove('Target')

    X = dataset[feature_names]
    y = dataset['Target']

    X_fair = fair_dataset[feature_names]
    y_fair = fair_dataset['Target']
    weights = fair_dataset['weights']

    # settiamo i nostri modelli sul dataset originale
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    # settiamo i nostri modelli sul dataset fair
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

    kf = KFold(n_splits=10)

    df_array = np.asarray(dataset)

    i = 0

    for train_index,test_index in kf.split(df_array):
        i = i + 1

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]
        weights_train = weights.iloc[train_index]

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]
        weights_test = weights.iloc[test_index]

        lr_model_pipeline.fit(X_train,y_train.values.ravel())
        rf_model_pipeline.fit(X_train,y_train.values.ravel())
        svm_model_pipeline.fit(X_train,y_train.values.ravel())
        xgb_model_pipeline.fit(X_train,y_train.values.ravel())

        validate(lr_model_pipeline, i, "std_models", 'lr', X_test, y_test)
        validate(rf_model_pipeline,i,'std_models','rf',X_test,y_test)
        validate(svm_model_pipeline,i,'std_models','svm',X_test,y_test)
        validate(xgb_model_pipeline,i,'std_models','xgb',X_test,y_test)

        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)
        xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel(),model__sample_weight=weights_train)

        validate(lr_fair_model_pipeline, i, "fair_models", 'lr', X_fair_test, y_fair_test)
        validate(rf_fair_model_pipeline,i,'fair_models','rf',X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,i,'fair_models','svm',X_fair_test,y_fair_test)
        validate(xgb_fair_model_pipeline,i,'fair_models','xgb',X_fair_test,y_fair_test)

        processed_train = processing_fairness(dataset,X_train,y_train,sensible_features_names,i)

        X_postop_train = processed_train[feature_names]
        y_postop_train = processed_train['Target']

        post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
        post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
        post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
        post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

        validate_postop(post_lr_model_pipeline,'lr',i,X_test,y_test)
        validate_postop(post_rf_model_pipeline,'rf',i,X_test,y_test)
        validate_postop(post_svm_model_pipeline,'svm',i,X_test,y_test)
        validate_postop(post_xgb_model_pipeline,'xgb',i,X_test,y_test)

    ## attuiamo una fase di postprocessing
    # test_postprocessing(lr_model_pipeline,'lr','std',dataset,X,sensible_features_names)
    # test_postprocessing(rf_model_pipeline,'rf','std',dataset,X,sensible_features_names)
    # test_postprocessing(svm_model_pipeline,'svm','std',dataset,X,sensible_features_names)
    # test_postprocessing(xgb_model_pipeline,'xgb','std',dataset,X,sensible_features_names)
    
    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_aif360_student_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_aif360_student_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_aif360_student_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_aif360_student_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_aif360_student_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_aif360_student_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_aif360_student_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_aif360_student_model.sav','wb'))

def processing_fairness(dataset,X_set,y_set,protected_features,index):

    fair_classifier = MetaFairClassifier(type='sr')

    train_dataset = pd.DataFrame(X_set)

    train_dataset['Target'] = y_set

    aif_train = BinaryLabelDataset(
        df=train_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'Gender': 1}]
    unprivileged_groups = [{'Gender': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    if index == 1:
        first_message = True
    else:
        first_message = False
    
    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Gender Mean difference pre inprocessing',first_message=first_message)
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Gender DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Gender Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Gender DI post inprocessing')

    privileged_groups = [{'Educational special needs': 0}]
    unprivileged_groups = [{'Educational special needs': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Sp. Needs Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Sp. Needs DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Sp. Needs Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Sp. Needs DI post inprocessing')

    privileged_groups = [{'International': 0}]
    unprivileged_groups = [{'International': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: International Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: International DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: International Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: International DI post inprocessing')

    mod_aif_dataset = StandardDataset(
        df=train_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    mod_aif_post = StandardDataset(
        df=fair_postop_df.convert_to_dataframe()[0],
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    age_aif_dataset = BinaryLabelDataset(
        df=mod_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    age_aif_post = BinaryLabelDataset(
        df=mod_aif_post.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'Age at enrollment': 1}]
    unprivileged_groups = [{'Age at enrollment': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=age_aif_dataset,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Age Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Age DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=age_aif_post,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Age Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Age DI post inprocessing')

    grade_mean = dataset['Admission grade'].mean()

    mod_aif_dataset = StandardDataset(
        df=train_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= grade_mean]
    )

    mod_aif_post = StandardDataset(
        df=fair_postop_df.convert_to_dataframe()[0],
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x < grade_mean]
    )

    grade_aif_dataset = BinaryLabelDataset(
        df=mod_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    grade_aif_post = BinaryLabelDataset(
        df=mod_aif_post.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'Admission grade': 1}]
    unprivileged_groups = [{'Admission grade': 0}]

    metrics_og = BinaryLabelDatasetMetric(dataset=grade_aif_dataset,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_postop_metrics(metrics_og.mean_difference(),f'{index} iter: Grade Mean difference pre inprocessing')
    print_postop_metrics(metrics_og.disparate_impact(),f'{index} iter: Grade DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=grade_aif_post,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_postop_metrics(metrics_trans.mean_difference(),f'{index} iter: Grade Mean difference post inprocessing')
    print_postop_metrics(metrics_trans.disparate_impact(),f'{index} iter: Grade DI post inprocessing')


    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train



def validate_postop(ml_model,model_type,index,X_test,y_test):
    pred = ml_model.predict(X_test)

    report = classification_report(y_pred=pred,y_true=y_test)

    if index == 1:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/postop_models/aif360/student/{model_type}_student_metrics_report.txt',open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write('Metriche di valutazione:')
        f.write(str(report))
        f.write('\n')

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
    with open(f"./reports/{model_vers}/aif360/student/{model_type}_student_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/aif360/student/{model_type}_student_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def test_fairness(dataset):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Gender']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    gender_privileged_group = [{'Gender': 1}]
    gender_unprivileged_group = [{'Gender': 0}]

    gender_metric_og = BinaryLabelDatasetMetric(dataset=aif_gender_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_og.mean_difference(),'Gender mean_difference before',first_message=True)
    print_fairness_metrics(gender_metric_og.disparate_impact(),"Gender DI before")
    
    gender_RW = Reweighing(unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    gender_trans_dataset = gender_RW.fit_transform(aif_gender_dataset)

    gender_metric_trans = BinaryLabelDatasetMetric(dataset=gender_trans_dataset,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_fairness_metrics(gender_metric_trans.mean_difference(),'Gender mean_difference after')
    print_fairness_metrics(gender_metric_trans.disparate_impact(),"Gender DI after")

    new_dataset = gender_trans_dataset.convert_to_dataframe()[0]
    sample_weights = gender_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # verifichiamo se gli studenti normodotati ricevono predizioni positive maggiori rispetto
    # agli studenti con disabilità
    sn_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
        instance_weights_name=['weights']
    )

    sn_privileged_group = [{'Educational special needs': 0}]
    sn_unprivileged_group = [{'Educational special needs': 1}]

    sn_metric_og = BinaryLabelDatasetMetric(dataset=sn_aif_dataset,unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    print_fairness_metrics(sn_metric_og.mean_difference(),'Special Needs mean_difference before')
    print_fairness_metrics(sn_metric_og.disparate_impact(),"Special Needs DI before")

    sn_RW = Reweighing(unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    sn_trans_dataset = sn_RW.fit_transform(sn_aif_dataset)

    sn_metric_trans = BinaryLabelDatasetMetric(dataset=sn_trans_dataset,unprivileged_groups=sn_unprivileged_group, privileged_groups=sn_privileged_group)

    print_fairness_metrics(sn_metric_trans.mean_difference(),'Special Needs mean_difference after')
    print_fairness_metrics(sn_metric_trans.disparate_impact(),"Special Needs DI after")

    # salviamo il nuovo dataset ottenuto e i pesi rivalutati
    new_dataset = sn_trans_dataset.convert_to_dataframe()[0]

    sample_weights = sn_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # valutiamo ora eventuale disparità nelle età degli studenti
    # cerchiamo di valutare se uno studente con meno di 30 anni sia favorito rispetto a studenti in età più avanzata
    std_age_aif_dataset = StandardDataset(
        df=new_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    mod_age_dataset = std_age_aif_dataset.convert_to_dataframe()[0]

    mod_age_dataset['weights'] = sample_weights

    age_aif_dataset = BinaryLabelDataset(
        df=std_age_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
        instance_weights_name=['weights'],
    )

    age_privileged_group = [{'Age at enrollment': 1}]
    age_unprivileged_group = [{'Age at enrollment': 0}]

    age_metric_og = BinaryLabelDatasetMetric(dataset=age_aif_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_og.mean_difference(),'Age mean_difference before')
    print_fairness_metrics(age_metric_og.disparate_impact(),"Age DI before")

    age_RW = Reweighing(unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    age_trans_dataset = age_RW.fit_transform(age_aif_dataset)

    age_metric_trans = BinaryLabelDatasetMetric(dataset=age_trans_dataset,unprivileged_groups=age_unprivileged_group,privileged_groups=age_privileged_group)

    print_fairness_metrics(age_metric_trans.mean_difference(),'Age mean_difference after')
    print_fairness_metrics(age_metric_trans.disparate_impact(),"Age DI after")

    new_dataset = age_trans_dataset.convert_to_dataframe()[0]

    sample_weights = age_trans_dataset.instance_weights

    new_dataset['weights'] = sample_weights

    # cerchiamo ora di stabilire se gli studenti internazionali sono svantaggiati
    int_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
        instance_weights_name=['weights'],
    )

    int_privileged_group = [{'International': 0}]
    int_unprivileged_group = [{'International': 1}]

    int_metric_og = BinaryLabelDatasetMetric(dataset=int_aif_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_og.mean_difference(),'International mean_difference before')
    print_fairness_metrics(int_metric_og.disparate_impact(),"International DI before")

    int_RW = Reweighing(unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    int_trans_dataset = int_RW.fit_transform(int_aif_dataset)

    int_metric_trans = BinaryLabelDatasetMetric(dataset=int_trans_dataset,unprivileged_groups=int_unprivileged_group,privileged_groups=int_privileged_group)

    print_fairness_metrics(int_metric_trans.mean_difference(),'International mean_difference after')
    print_fairness_metrics(int_metric_trans.disparate_impact(),"International DI after")

    new_dataset = int_trans_dataset.convert_to_dataframe()[0]

    sample_weights = int_trans_dataset.instance_weights
    
    new_dataset['weights'] = sample_weights

    # con questo comando stampiamo il voto più alto e più basso ottenuti all'interno del campione utilizzato.
    # Otteniamo anche la media, utilizzeremo la media aritmentica dei voti come valore per dividere il dataset in studenti con voto sotto la media
    # e sopra la media.
    # print(new_dataset['Admission grade'].mean(), new_dataset['Admission grade'].max(), new_dataset['Admission grade'].min())
    mean_grade = new_dataset['Admission grade'].mean()
    
    # infine, cerchiamo di valutare se gli studenti con un basso voto di ammissione siano sfavoriti per predizione positiva
    # selezioniamo appunto come favoriti tutti gli studenti il cui voto di ammissione supera il voto medio di ammissione
    std_grade_dataset = StandardDataset(
        df=new_dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= mean_grade]
    )

    new_dataset = std_grade_dataset.convert_to_dataframe()[0]

    new_dataset['weights'] = sample_weights

    # costruiamo dataset binario per testare fairness della nostra scelta
    grade_aif_dataset = BinaryLabelDataset(
        df=new_dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
        instance_weights_name=['weights'],
    )

    grade_privileged_group = [{'Admission grade': 1}]
    grade_unprivileged_group = [{'Admission grade': 0}]

    grade_metric_og = BinaryLabelDatasetMetric(dataset=grade_aif_dataset,unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_og.mean_difference(),'Grade mean_difference before')
    print_fairness_metrics(grade_metric_og.disparate_impact(),"Grade DI before")

    grade_RW = Reweighing(unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    grade_trans_dataset = grade_RW.fit_transform(grade_aif_dataset)

    grade_metric_trans = BinaryLabelDatasetMetric(dataset=grade_trans_dataset, unprivileged_groups=grade_unprivileged_group, privileged_groups=grade_privileged_group)

    print_fairness_metrics(grade_metric_trans.mean_difference(),'Grade mean_difference after')
    print_fairness_metrics(grade_metric_trans.disparate_impact(),"Grade DI after")

    # restituiamo infine i pesi ottenuti da quest'ultima iterazione
    sample_weights = grade_trans_dataset.instance_weights

    return sample_weights


def print_fairness_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/student_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

def print_postop_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/postprocessing/aif360/student_report.txt",open_type) as f:
        f.write(f"{message}: {metric}")
        f.write('\n')

load_dataset()