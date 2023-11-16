import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
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

    # addestriamo più modelli standard sul dataset originale
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

    model_URL = "https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-classification/versions/2"
    resnet_google = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1./255, input_shape=(48,48, 3)),
            hub.KerasLayer(model_URL),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    resnet_google.compile(loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2),'precision','recall'])

    resnet_history = resnet_google.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size,
    )

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['f1_score'])
    plt.title('model F1_Score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.savefig('./figs/std/std_resnet_f1.png')

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('./figs/std/std_resnet_accuracy.png')

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.savefig('./figs/std/std_resnet_precision.png')

    plt.figure(figsize=(20,8))
    plt.plot(resnet_history.history['recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.savefig('./figs/std/std_resnet_recall.png')

    resnet_loss, resnet_accuracy, resnet_f1, resnet_precision, resnet_recall = resnet_google.evaluate(validation_generator)

    with open('./reports/std_models/resnet_gender_recognition_report.txt','w') as f:
        f.write('ResnetV2 model\n')
        f.write(f"Accuracy: {round(resnet_accuracy,3)}\n")
        f.write(f'F1 Score: {round(resnet_f1,3)}\n')
        f.write(f'Precision: {round(resnet_precision,3)}\n')
        f.write(f'Recall: {round(resnet_recall,3)}\n')

    m_json = resnet_google.to_json()
    with open('./output_models/std_models/resnet_model/resnet_gender_recognition_model.json','w') as f:
        f.write(m_json)

    resnet_google.save_weights('./output_models/std_models/resnet_model/resnet_std_weights.h5')

def print_time(time):
    with open('./reports/time_reports/gender/std_resnet_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)