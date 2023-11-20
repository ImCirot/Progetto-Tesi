import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow_addons as tfa

#
#
#
# IMPORTANTE: Per poter eseguire questo codice, scaricare il dataset UTKFace disponibile al seguente link: https://susanqq.github.io/UTKFace/
# scaricare tutte e 3 le parti, creare una cartella denominata "UTKFace Dataset" all'interno della directory generale del progetto
#
#
#

@track_emissions(country_iso_code='ITA',offline=True)
def load_dataset():
    ## funzione di creazione del dataset e setting del modello

    # setting del percorso contenente le immagini del dataset
    folders = ["./UTKFace Dataset/"]

    # settiamo un dizionario per contenere il numero di istanze delle due possibili categorie
    # ("male" e "female")
    countCat = {0:0, 1:0}

    # settiamo un dizionario per contenere i "pesi" di ciascuna possibile classe
    class_weight = {0:1, 1:1}

    # settiamo delle liste per contenere i valori degli attributi realtivi ad ogni immagine
    data, age_label, gender_label, race_label = [], [], [], []

    # ciclo per estrarre informazioni dalle immagini e costruire il dataset contenente nome, sesso, età e razza di ogni indivudio presente
    for folder in folders:
        for file in glob.glob(folder+"*.jpg"):
            file = file.replace(folder, "")
            age, gender, race = file.split("_")[0:3]
            age, gender, race = int(age), int(gender), int(race)
            countCat[gender]+=1
            data.append(folder + file)
            age_label.append(str(age))
            gender_label.append(str(gender))
            race_label.append(str(race))

    # settiamo il numero di sample totali
    n_samples = len(data)
    
    # settiamo il numero di classi
    n_class = 2

    # ciclo che permette di calcolare i pesi di ogni classe seguendo l'equazione: num. totale samples/(numero di classi * numero di sample della classe)
    for key in class_weight:
        class_weight[key] = n_samples/(n_class*countCat[key])
    
    # creiamo un dataframe contenente il nome del file, l'età, genere e razza di ogni sample all'interno del dataset
    std_df = pd.DataFrame(data={"filename": data, "age": age_label, "gender": gender_label, 'race': race_label})

    # valutiamo la postprocessing modello sul dataset 
    postop_model(std_df)

def postop_model(df):
    ## funzione di postprocessing dei risultati dle modello

    # setting dimensioni immagine
    image_size = (48, 48)
    
    # setting dimensione del batch per ogni iterazione
    batch_size = 64

    # settiamo l'oggetto offerto da TensorFLow che ci permette di caricare immagini e creare un dataset su quest'ultime
    # settiamo la divisione come 80/20 training/testing come standard
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
    

    # creiamo il dataset di training del modello
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        y_col='gender',
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
    )

    # creiamo il dataset di testing del modello
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
    )

    features = df.columns.tolist()
    features.remove('gender') 

    df_train = df[df['filename'].isin(train_generator.filenames)]
    df_test = df[df['filename'].isin(validation_generator.filenames)]

    X_train = df_train[features]
    y_train = df_train['gender'].astype(int)

    X_test = df_test[features]
    y_test = df_test['gender'].astype(int)

    json_file = open('./output_models/std_models/resnet_model/resnet_gender_recognition_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights('./output_models/std_models/resnet_model/resnet_std_weights.h5')

    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    model.compile(loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    pred = model.predict(validation_generator)
    pred = np.argmax(pred,axis=1)
    std_pred = pd.DataFrame(pred,columns=['gender'])
    std_pred[features] = df_test[features]

    fair_pred = test_fairness(df_test,std_pred)

    resnet_accuracy = accuracy_score(y_true=y_test,y_pred=fair_pred['gender'])

    resnet_f1_score = f1_score(y_true=y_test,y_pred=fair_pred['gender'])

    resnet_precision_score = precision_score(y_true=y_test,y_pred=fair_pred['gender'])

    resnet_recall_score = recall_score(y_true=y_test,y_pred=fair_pred['gender'])
    
    with open('./reports/postprocessing_models/resnet_gender_recognition_report.txt','w') as f:
        f.write('ResnetV2 model\n')
        f.write(f"Accuracy: {round(resnet_accuracy,3)}\n")
        f.write(f"F1 score: {round(resnet_f1_score,3)}\n")
        f.write(f"Precision: {round(resnet_precision_score,3)}\n")
        f.write(f"Recall: {round(resnet_recall_score,3)}\n")

def test_fairness(dataset,pred):
    ## funzione che calcola alcune metriche di fairness e cerca di mitigare eventuali discriminazioni presenti nel dataset

    dataset = dataset.drop('filename',axis=1)
    pred = pred.drop('filename',axis=1)

    dataset= dataset.astype(int)
    pred = pred.astype(int)

    race_aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=0,
        unfavorable_label=1,
        label_names=['gender'],
        protected_attribute_names=['race'],
    )

    race_aif_pred = BinaryLabelDataset(
        df=pred,
        favorable_label=0,
        unfavorable_label=1,
        label_names=['gender'],
        protected_attribute_names=['race'],
    )

    race_privileged_groups = [{'race': 0},{'race':2},{'race':3}]
    race_unprivileged_groups = [{'race': 1},{'race':4}]

    race_metric_original = BinaryLabelDatasetMetric(dataset=race_aif_dataset, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)
    agg_metrics_og = ClassificationMetric(dataset=race_aif_dataset,classified_dataset=race_aif_pred,privileged_groups=race_privileged_groups,unprivileged_groups=race_unprivileged_groups)

    print_metrics('mean_difference before', race_metric_original.mean_difference(),first_message=True)
    print_metrics('DI before', race_metric_original.disparate_impact())
    print_metrics('Eq. odds diff before', agg_metrics_og.equal_opportunity_difference())

    eqoods = CalibratedEqOddsPostprocessing(unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups,cost_constraint='fpr',seed=42)

    race_transformed = eqoods.fit_predict(race_aif_dataset,race_aif_pred,threshold=0.8)

    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_transformed, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)
    agg_metrics_trans = ClassificationMetric(dataset=race_aif_dataset,classified_dataset=race_transformed,privileged_groups=race_privileged_groups,unprivileged_groups=race_unprivileged_groups)

    print_metrics('mean_difference after', race_metric_transformed.mean_difference())
    print_metrics('DI after', race_metric_transformed.disparate_impact())
    print_metrics('Eq. odds diff after', agg_metrics_trans.equal_opportunity_difference())

    fair_dataset = race_transformed.convert_to_dataframe()[0]

    fair_dataset = fair_dataset.astype(int)

    return fair_dataset

def print_metrics(message,metric,first_message=False):
    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/fairness_reports/postprocessing/aif360/resnet_gender_report.txt',open_type) as f:
        f.write(f'{message}: {round(metric,3)}\n')

def print_time(time):
    with open('./reports/time_reports/gender/aif360/resnet_postprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)