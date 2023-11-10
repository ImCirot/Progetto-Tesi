import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from datetime import datetime

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

    # addestriamo più modelli fair sul dataset modificato
    training_and_testing_model(std_df)

def training_and_testing_model(df):
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

    inprocessing_df = inprocess_op(df)
    filenames = df['filename'].tolist()
    inprocessing_df['filename'] = filenames

    # creiamo il dataset di training del modello
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=inprocessing_df,
        y_col='gender',
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
    )

    # creiamo il dataset di testing del modello
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=inprocessing_df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
    )

    model_URL = "https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-classification/versions/2"
    resnet_google = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1./255, input_shape=(48,48, 3)),
            hub.KerasLayer(model_URL),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    resnet_google.compile(loss='categorical_crossentropy', metrics=['accuracy','AUC'])

    model_name = "resnet_model.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_best_only = True,
        verbose=1,
        filepath=f'./output_models/inprocessing_models/{model_name}'
    )

    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',min_delta = 0, patience = 5,
        verbose = 1, restore_best_weights=True
        )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.0001)
    

    resnet_history = resnet_google.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size,
        callbacks=[checkpoint,reduce_lr]
    )

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['auc'])
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.savefig('./figs/inprocessing_resnet_roc-auc.png')

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('./figs/inprocessing_resnet_accuracy.png')

    resnet_loss, resnet_accuracy, resnet_auc = resnet_google.evaluate(validation_generator)

    with open('./reports/inprocessing_models/resnet_gender_recognition_report.txt','w') as f:
        f.write('ResnetV2 model\n')
        f.write(f"Accuracy: {round(resnet_accuracy,3)}\n")
        f.write(f'AUC-ROC: {round(resnet_auc,3)}\n')

def inprocess_op(dataset):
    ## funzione che calcola alcune metriche di fairness e cerca di mitigare eventuali discriminazioni presenti nel dataset

    fair_classifier = MetaFairClassifier(type='sr',seed=42)

    dataset = dataset.drop('filename',axis=1)

    dataset= dataset.astype(int)

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
    print_metrics('mean_difference before', race_metric_original.mean_difference(),first_message=True)
    print_metrics('DI before', race_metric_original.disparate_impact())

    race_transformed = fair_classifier.fit_predict(race_aif_dataset)

    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_transformed, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)

    print_metrics('mean_difference after', race_metric_transformed.mean_difference())
    print_metrics('DI after', race_metric_transformed.disparate_impact())

    fair_dataset = race_transformed.convert_to_dataframe()[0]

    fair_dataset = fair_dataset.astype(int)
    fair_dataset = fair_dataset.astype(str)

    return fair_dataset

def print_metrics(message,metric,first_message=False):
    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/fairness_reports/inprocessing/resnet_gender_report.txt',open_type) as f:
        f.write(f'{message}: {round(metric,3)}\n')

def print_time(time):
    with open('./reports/time_reports/gender/resnet_inprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)