import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob

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
    folders = ["./UTKFace Dataset/"]
    countCat = {0:0, 1:0}
    class_weight = {0:1, 1:1}
    data, age_label, gender_label, race_label = [], [], [], []

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
    n_samples = len(data)
    n_class = 2
    for key in class_weight:
        class_weight[key] = n_samples/(n_class*countCat[key])
    
    std_df = pd.DataFrame(data={"filename": data, "age": age_label, "gender": gender_label, 'race': race_label})

    # fair_df = test_fairness(std_df)

    training_and_testing_model(std_df,'std',class_weight)

    # training_and_testing_model(fair_df,'fair',class_weight)

def training_and_testing_model(df,df_type,class_weight):

    image_size = (48, 48)

    batch_size = 32

    epochs = 15

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        y_col='gender',
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical'
    )

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'
    )
        
    classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=True, 
        weights=None, 
        input_tensor=None,
        input_shape=image_size + (3,), 
        pooling=None, 
        classes=2
    )
    
    classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size, 
        class_weight=class_weight
    )

    classifier.save(f"./models/{df_type}_gender_recognition_model")

load_dataset()