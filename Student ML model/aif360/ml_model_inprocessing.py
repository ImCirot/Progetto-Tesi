from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from codecarbon import track_emissions
import xgboost as xgb
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.inprocessing import *
from fairlearn.reductions import DemographicParity
import pickle
from datetime import datetime
from time import sleep

def load_dataset():
    ## funzione di load del dataset dal file csv

    # carica il dataset dal file csv
    df = pd.read_csv('./Student Dataset/dataset.csv')

    # drop ID dal dataframe
    df.drop('ID', inplace=True,axis=1)

    # richiamo funzione di training e testing dei modelli
    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_testing_models(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(120)
            print('########################### IDLE TIME FINISH ###########################')

@track_emissions(country_iso_code='ITA',offline=True)
def training_testing_models(dataset):
    ## funzione di training e testing dei vari modelli

    # setting feature sensibili
    sensible_features_names = [
        'Gender','Educational special needs',
        'Age at enrollment','Admission grade', 'International',
    ]

    feature_names = dataset.columns.tolist()

    feature_names.remove('Target')

    X = dataset[feature_names]
    y = dataset['Target']

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_student_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_student_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_student_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_student_model.sav','rb'))

    post_lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    post_rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    post_svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    post_xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic',random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    processed_train = processing_fairness(dataset,X_train,y_train,sensible_features_names)

    X_postop_train = processed_train[feature_names]
    y_postop_train = processed_train['Target']

    print(f'######### Training modelli #########')
    post_lr_model_pipeline.fit(X_postop_train,y_postop_train)
    post_rf_model_pipeline.fit(X_postop_train,y_postop_train)
    post_svm_model_pipeline.fit(X_postop_train,y_postop_train)
    post_xgb_model_pipeline.fit(X_postop_train,y_postop_train)

    print(f'######### Testing modelli #########')
    validate(post_lr_model_pipeline,'lr',X_test,y_test,True)
    validate(post_rf_model_pipeline,'rf',X_test,y_test)
    validate(post_svm_model_pipeline,'svm',X_test,y_test)
    validate(post_xgb_model_pipeline,'xgb',X_test,y_test)

    print(f'######### Testing Fairness #########')
    X_test_df = X_test.copy(deep=True)
    X_test_df['Target'] = y_test

    lr_inproc_pred = X_test.copy(deep=True)
    lr_inproc_pred['Target'] = post_lr_model_pipeline.predict(X_test)

    rf_inproc_pred =  X_test.copy(deep=True)
    rf_inproc_pred['Target'] = post_rf_model_pipeline.predict(X_test)

    svm_inproc_pred =  X_test.copy(deep=True)
    svm_inproc_pred['Target'] = post_svm_model_pipeline.predict(X_test)

    xgb_inproc_pred =  X_test.copy(deep=True)
    xgb_inproc_pred['Target'] = post_xgb_model_pipeline.predict(X_test)

    lr_pred = X_test.copy(deep=True)
    lr_pred['Target'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test.copy(deep=True)
    rf_pred['Target'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test.copy(deep=True)
    svm_pred['Target'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test.copy(deep=True)
    xgb_pred['Target'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,'lr')
    eq_odds_fair_report(X_test_df,rf_pred,'rf')
    eq_odds_fair_report(X_test_df,svm_pred,'svm')
    eq_odds_fair_report(X_test_df,xgb_pred,'xgb')

    eq_odds_fair_report(X_test_df,lr_inproc_pred,'lr_inprocessing')
    eq_odds_fair_report(X_test_df,rf_inproc_pred,'rf_inprocessing')
    eq_odds_fair_report(X_test_df,svm_inproc_pred,'svm_inprocessing')
    eq_odds_fair_report(X_test_df,xgb_inproc_pred,'xgb_inprocessing')

    
    print(f'######### Salvataggio modelli #########')
    pickle.dump(post_lr_model_pipeline,open('./output_models/inprocess_models/lr_aif360_student_model.sav','wb'))
    pickle.dump(post_rf_model_pipeline,open('./output_models/inprocess_models/rf_aif360_student_model.sav','wb'))
    pickle.dump(post_svm_model_pipeline,open('./output_models/inprocess_models/svm_aif360_student_model.sav','wb'))
    pickle.dump(post_xgb_model_pipeline,open('./output_models/inprocess_models/xgb_aif360_student_model.sav','wb'))

    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def eq_odds_fair_report(dataset,prediction,name):
    ## funzione che testa fairness del dataset sulla base degli attributi sensibili

    aif_gender_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Gender']
    )

    aif_gender_prediction = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Gender']
    )

    # cerchiamo di stabilire se le donne sono o meno svantaggiate nella predizione positiva
    gender_privileged_group = [{'Gender': 1}]
    gender_unprivileged_group = [{'Gender': 0}]

    metrics = ClassificationMetric(dataset=aif_gender_dataset,classified_dataset=aif_gender_prediction,unprivileged_groups=gender_unprivileged_group,privileged_groups=gender_privileged_group)

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Gender Eq. Odds difference')

    # verifichiamo se gli studenti normodotati ricevono predizioni positive maggiori rispetto
    # agli studenti con disabilità
    sn_aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
    )

    sn_aif_prediction = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Educational special needs'],
    )

    sn_privileged_group = [{'Educational special needs': 0}]
    sn_unprivileged_group = [{'Educational special needs': 1}]

    metrics = ClassificationMetric(dataset=sn_aif_dataset,classified_dataset=sn_aif_prediction,unprivileged_groups=sn_unprivileged_group,privileged_groups=sn_privileged_group)

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Special Needs Eq. Odds difference')


    # valutiamo ora eventuale disparità nelle età degli studenti
    # cerchiamo di valutare se uno studente con meno di 30 anni sia favorito rispetto a studenti in età più avanzata
    std_age_aif_dataset = StandardDataset(
        df=dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    std_age_aif_prediction = StandardDataset(
        df=prediction,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Age at enrollment'],
        privileged_classes=[lambda x: x <= 30]
    )

    age_aif_dataset = BinaryLabelDataset(
        df=std_age_aif_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
    )

    age_aif_prediction = BinaryLabelDataset(
        df=std_age_aif_prediction.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Age at enrollment'],
    )

    age_privileged_group = [{'Age at enrollment': 1}]
    age_unprivileged_group = [{'Age at enrollment': 0}]

    metrics = ClassificationMetric(dataset=age_aif_dataset,classified_dataset=age_aif_prediction,privileged_groups=age_privileged_group,unprivileged_groups=age_unprivileged_group)

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Age Eq. Odds difference')

    # cerchiamo ora di stabilire se gli studenti internazionali sono svantaggiati
    int_aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
    )

    int_aif_prediction = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['International'],
    )

    int_privileged_group = [{'International': 0}]
    int_unprivileged_group = [{'International': 1}]

    metrics = ClassificationMetric(dataset=int_aif_dataset,classified_dataset=int_aif_prediction,privileged_groups=int_privileged_group,unprivileged_groups=int_unprivileged_group)

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model International Eq. Odds difference')

    mean_grade = dataset['Admission grade'].mean()
    
    # infine, cerchiamo di valutare se gli studenti con un basso voto di ammissione siano sfavoriti per predizione positiva
    # selezioniamo appunto come favoriti tutti gli studenti il cui voto di ammissione supera il voto medio di ammissione
    std_grade_dataset = StandardDataset(
        df=dataset,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= mean_grade]
    )

    std_grade_prediction = StandardDataset(
        df=prediction,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=['Admission grade'],
        privileged_classes=[lambda x: x >= mean_grade]
    )


    # costruiamo dataset binario per testare fairness della nostra scelta
    grade_aif_dataset = BinaryLabelDataset(
        df=std_grade_dataset.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
    )

    grade_aif_prediction = BinaryLabelDataset(
        df=std_grade_prediction.convert_to_dataframe()[0],
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=['Admission grade'],
    )

    grade_privileged_group = [{'Admission grade': 1}]
    grade_unprivileged_group = [{'Admission grade': 0}]

    metrics = ClassificationMetric(dataset=grade_aif_dataset,classified_dataset=grade_aif_prediction,unprivileged_groups=grade_unprivileged_group,privileged_groups=grade_privileged_group)

    print_metrics((metrics.true_positive_rate_difference() - metrics.false_positive_rate_difference()),f'{name}_model Grade Eq. Odds difference')

def processing_fairness(dataset,X_set,y_set,protected_features):

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
    
    print_metrics(metrics_og.mean_difference(),f'Gender Mean difference pre inprocessing',first_message=True)
    print_metrics(metrics_og.disparate_impact(),f'Gender DI pre inprocessing')

    fair_postop_df = fair_classifier.fit_predict(dataset=aif_train)

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Gender Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Gender DI post inprocessing')

    privileged_groups = [{'Educational special needs': 0}]
    unprivileged_groups = [{'Educational special needs': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_metrics(metrics_og.mean_difference(),f'Sp. Needs Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Sp. Needs DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Sp. Needs Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Sp. Needs DI post inprocessing')

    privileged_groups = [{'International': 0}]
    unprivileged_groups = [{'International': 1}]

    metrics_og = BinaryLabelDatasetMetric(dataset=aif_train,privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups)

    print_metrics(metrics_og.mean_difference(),f'International Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'International DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=fair_postop_df,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'International Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'International DI post inprocessing')

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

    print_metrics(metrics_og.mean_difference(),f'Age Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Age DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=age_aif_post,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Age Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Age DI post inprocessing')

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

    print_metrics(metrics_og.mean_difference(),f'Grade Mean difference pre inprocessing')
    print_metrics(metrics_og.disparate_impact(),f'Grade DI pre inprocessing')

    metrics_trans = BinaryLabelDatasetMetric(dataset=grade_aif_post,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print_metrics(metrics_trans.mean_difference(),f'Grade Mean difference post inprocessing')
    print_metrics(metrics_trans.disparate_impact(),f'Grade DI post inprocessing')


    postop_train = fair_postop_df.convert_to_dataframe()[0]
    
    return postop_train

def validate(ml_model,model_type,X_test,y_test,first=False):
    pred = ml_model.predict(X_test)

    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/inprocessing_models/aif360/student_metrics_report.txt',open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def print_metrics(metric, message, first_message=False):
    ## funzione per stampare in file le metriche di fairness del modello passato in input

    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'
    
    #scriviamo su un file la metrica passata
    with open(f"./reports/fairness_reports/inprocessing/aif360/student_report.txt",open_type) as f:
        f.write(f"{message}: {round(metric,3)}")
        f.write('\n')

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/aif360/student_inprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)