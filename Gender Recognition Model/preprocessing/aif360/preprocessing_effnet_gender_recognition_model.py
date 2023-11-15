import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
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

    # effettuiamo delle operazioni di fairness e otteniamo un dataset ¨fair"
    (fair_df,sample_weights) = test_fairness(std_df)
    fair_df['weights'] = sample_weights
    filenames = std_df['filename'].tolist()
    fair_df['filename'] = filenames

    # addestriamo più modelli fair sul dataset modificato
    training_and_testing_model(std_df,fair_df)

def training_and_testing_model(std_df,fair_df):
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
    fair_train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

    # creiamo il dataset di training del modello
    fair_train_generator = fair_train_datagen.flow_from_dataframe(
        dataframe=fair_df,
        y_col='gender',
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
        weight_col='weights'
    )

    # creiamo il dataset di testing del modello
    fair_validation_generator = fair_train_datagen.flow_from_dataframe(
        dataframe=fair_df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
        weight_col='weights'
    )

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

    # creiamo il dataset di training del modello
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=std_df,
        y_col='gender',
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
    )

    # creiamo il dataset di testing del modello
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=std_df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
    )

    model_URL = "https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b7-classification/versions/1"
    effnet_model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1./255, input_shape=(48,48, 3)),
            hub.KerasLayer(model_URL),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    effnet_model.compile(loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2)])
    
    # addestriamo il modello EfficientNet
    effnet_history = effnet_model.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples//batch_size,
    )

    plt.figure(figsize=(20,8))
    plt.plot(effnet_history.history['f1_score'])
    plt.title('model F1 Score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.savefig('./figs/preprocessing_effnet_f1.png')

    plt.figure(figsize=(20,8))
    plt.plot(effnet_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('./figs/preprocessing_effnet_accuracy.png')

    effnet_loss, effnet_accuracy, effnet_auc = effnet_model.evaluate(validation_generator)

    with open('./reports/preprocessing_models/aif360/effnet_gender_recognition_report.txt','w') as f:
        f.write('EfficentNet Model\n')
        f.write(f"Accuracy: {round(effnet_accuracy,3)}\n")
        f.write(f'AUC-ROC: {round(effnet_auc,3)}\n')

    m_json = effnet_model.to_json()
    with open('./output_models/preprocessing_models/effnet_model/fairlearn/effnet_gender_recognition_model.json','w') as f:
        f.write(m_json)

    effnet_model.save_weights('./output_models/preprocessing_models/effnet_model/fairlearn/effnet_std_weights.h5')

    json_file = open('./output_models/std_models/effnet_model/effnet_gender_recognition_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json,custom_objects={'Rescaling':tf.keras.layers.Rescaling,'KerasLayer':hub.KerasLayer})
    model.load_weights('./output_models/std_models/effnet_model/effnet_std_weights.h5')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2)])

    features = std_df.columns.tolist()
    features.remove('gender') 

    df_fair_train = fair_df[fair_df['filename'].isin(fair_train_generator.filenames)]
    df_fair_test = fair_df[fair_df['filename'].isin(fair_validation_generator.filenames)]
    df_fair_test = df_fair_test.drop('weights',axis=1)
    
    df_std_train = std_df[std_df['filename'].isin(train_generator.filenames)]
    df_std_test = std_df[std_df['filename'].isin(validation_generator.filenames)]

    pred = model.predict(validation_generator)
    pred = np.argmax(pred,axis=1)
    std_pred = pd.DataFrame(pred,columns=['gender'])
    std_pred[features] = df_std_test[features]

    pred = effnet_model.predict(validation_generator)
    pred = np.argmax(pred,axis=1)
    fair_pred = pd.DataFrame(df_fair_test,columns=features)
    fair_pred['gender'] = pred
    fairness_op(df_std_test,std_pred,'std')

    fairness_op(df_fair_test,fair_pred,'fair')

def fairness_op(dataset,pred,name):
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

    race_metric_original = ClassificationMetric(dataset=race_aif_dataset,classified_dataset=race_aif_pred, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)
    print_metrics(f'{name}_model Eq. Odds diff', race_metric_original.true_positive_rate_difference()-race_metric_original.false_positive_rate_difference())

def test_fairness(dataset):
    ## funzione che calcola alcune metriche di fairness e cerca di mitigare eventuali discriminazioni presenti nel dataset

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

    RACE_RW = Reweighing(unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)

    race_transformed = RACE_RW.fit_transform(race_aif_dataset)

    race_metric_transformed = BinaryLabelDatasetMetric(dataset=race_transformed, privileged_groups=race_privileged_groups, unprivileged_groups=race_unprivileged_groups)

    print_metrics('mean_difference after', race_metric_transformed.mean_difference())
    print_metrics('DI after', race_metric_transformed.disparate_impact())

    fair_dataset = race_transformed.convert_to_dataframe()[0]

    fair_dataset = fair_dataset.astype(int)
    fair_dataset = fair_dataset.astype(str)

    return (fair_dataset,race_transformed.instance_weights)

def print_metrics(message,metric,first_message=False):
    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/fairness_reports/preprocessing/aif360/effnet_gender_report.txt',open_type) as f:
        f.write(f'{message}: {round(metric,3)}\n')

def print_time(time):
    with open('./reports/time_reports/gender/effnet_fair_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)