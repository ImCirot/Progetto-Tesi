import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset

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

    print(f"num. of males: {std_df[std_df['gender'] == 0].shape[0]}")
    print(f"num. of females: {std_df[std_df['gender'] == 1].shape[0]}")
    white_ind = std_df[std_df['race'] == 0]
    non_white_ind = std_df[std_df['race'] != 0]
    num_white_males = white_ind[white_ind['gender'] == 0].shape[0]
    num_non_white_males = non_white_ind[non_white_ind['gender'] == 0].shape[0]
    print(f'white male: {num_white_males}')
    print(f'non white male: {num_non_white_males}')

    # effettuiamo delle operazioni di fairness e otteniamo un dataset ¨fair"
    fair_df = test_fairness(std_df)
    filenames = std_df['filename'].tolist()
    fair_df['filename'] = filenames

    # addestriamo più modelli standard sul dataset originale
    training_and_testing_model(std_df,'std',class_weight)

    # addestriamo più modelli fair sul dataset modificato
    training_and_testing_model(fair_df,'fair',class_weight)

def training_and_testing_model(df,df_type,class_weight):
    ## funzione di apprendimento e validazione del modello

    # setting dimensioni immagine
    image_size = (48, 48)
    
    # setting dimensione del batch per ogni iterazione
    batch_size = 64
    
    # setting numero di epoche richieste
    # (numero di iterazioni per cui viene ripetuto training e testing del modello)
    epochs = 15

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
        class_mode='categorical'
    )

    # creiamo il dataset di testing del modello
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'
    )
    
    # creiamo il modello sfruttando la strategia MobileNetV2 offerta da TensorFlow
    mn_classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=True, 
        weights=None, 
        input_tensor=None,
        input_shape=image_size + (3,), 
        pooling=None, 
        classes=2
    )

    # creiamo il modello sfruttando la strategia ResNet50 offerta da TensorFlow
    resnet_classifier = tf.keras.applications.resnet50.ResNet50(
        include_top=True, 
        weights=None, 
        input_tensor=None,
        input_shape=image_size + (3,), 
        pooling=None, 
        classes=2
    )
    
    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    mn_classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    resnet_classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # addestriamo il modello MobileNetV2
    mn_classifier.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size, 
        class_weight=class_weight
    )

    resnet_classifier.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=5, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size, 
        class_weight=class_weight
    )

    # salviamo il modello in una directory di output per usi futuri
    mn_classifier.save(f"./output_models/{df_type}_MobileNetV2_gender_recognition_model")

    resnet_classifier.save(f'./output_models/{df_type}_ResNet50_gender_recognition_model')

def test_fairness(dataset):
    ## funzione che calcola alcune metriche di fairness e cerca di mitigare eventuali discriminazioni presenti nel dataset

    dataset = dataset.drop('filename',axis=1)

    dataset= dataset.astype(int)
    print(dataset.head)

    race_aif_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=0,
        unfavorable_label=1,
        label_names=['gender'],
        protected_attribute_names=['race'],
    )

    race_privileged_groups = [{'race': 0},{'race':2},{'race':3}]
    race_unprivileged_groups = [{'race': 1},{'race':4}]

    race_metric_original = BinaryLabelDatasetMetric(dataset=race_aif_dataset, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)
    print(f'num. of priviliged: {race_metric_original.num_positives(privileged=True)}')
    print(f'num. of unpriv: {race_metric_original.num_positives(privileged=False)}')
    print(f'mean_difference before: {race_metric_original.mean_difference()}')

    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)

    race_transformed = RACE_RW.fit_transform(race_aif_dataset)

    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_transformed, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)

    print(f'num. of priviliged: {race_metric_transformed.num_positives(privileged=True)}')
    print(f'num. of unpriv: {race_metric_transformed.num_positives(privileged=False)}')
    print(f'mean_difference after: {race_metric_transformed.mean_difference()}')

    fair_dataset = race_transformed.convert_to_dataframe()[0]

    fair_dataset = fair_dataset.astype(int)
    fair_dataset = fair_dataset.astype(str)

    return fair_dataset

load_dataset()