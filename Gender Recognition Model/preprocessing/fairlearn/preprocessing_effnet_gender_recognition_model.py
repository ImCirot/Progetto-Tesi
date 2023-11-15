import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from codecarbon import track_emissions
import glob
from fairlearn.metrics import demographic_parity_difference,demographic_parity_ratio,equalized_odds_difference
from fairlearn.preprocessing import CorrelationRemover
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
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
    fair_df = test_fairness(std_df)
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
    )

    # creiamo il dataset di testing del modello
    fair_validaton_generator = fair_train_datagen.flow_from_dataframe(
        dataframe=fair_df,
        y_col="gender",
        shuffle=True,
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
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
    validaton_generator = train_datagen.flow_from_dataframe(
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

    effnet_history = effnet_model.fit(
        fair_train_generator, 
        steps_per_epoch=fair_train_generator.samples//batch_size, 
        epochs=epochs, 
        validation_data=fair_validaton_generator, 
        validation_steps=fair_validaton_generator.samples//batch_size,
    )

    plt.figure(figsize=(20,8))
    plt.plot(effnet_history.history['f1_score'])
    plt.title('model F1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.savefig('./figs/fairlearn/preprocessing_effnet_f1.png')

    plt.figure(figsize=(20,8))
    plt.plot(effnet_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('./figs/fairlearn/preprocessing_effnet_accuracy.png')

    effnet_loss, effnet_accuracy, effnet_auc = effnet_model.evaluate(fair_validaton_generator)

    with open('./reports/preprocessing_models/fairlearn/effnet_gender_recognition_report.txt','w') as f:
        f.write('Effnet model\n')
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

    # indichiamo ai modello di stabilire il proprio comportamento su accuracy e categorical_crossentropy
    model.compile(loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2)])

    features = std_df.columns.tolist()
    features.remove('gender') 

    df_fair_train = fair_df[fair_df['filename'].isin(fair_train_generator.filenames)]
    df_fair_test = fair_df[fair_df['filename'].isin(fair_validaton_generator.filenames)]
    
    df_std_train = std_df[std_df['filename'].isin(train_generator.filenames)]
    df_std_test = std_df[std_df['filename'].isin(validaton_generator.filenames)]

    target = ['gender']
    prot_attrb = ['race']

    X_fair_train = df_fair_train[features]
    y_fair_train = df_fair_train[target].astype(int)

    X_fair_test = df_fair_test[features]
    y_fair_test = df_fair_test[target].astype(int)

    X_train = df_std_train[features]
    y_train = df_std_train[target].astype(int)

    X_test = df_std_test[features]
    y_test = df_std_test[target].astype(int)

    pred = model.predict(validaton_generator)
    print(pred)
    pred = np.argmax(pred,axis=1)
    print(pred)
    std_pred = pd.DataFrame(pred,columns=[target])
    std_pred[features] = df_std_test[features]
    print(std_pred)

    pred = effnet_model.predict(validaton_generator)
    print(pred)
    pred = np.argmax(pred,axis=1)
    print(pred)
    fair_pred = pd.DataFrame(pred,columns=[target])
    fair_pred[features] = df_fair_test[features]
    print(fair_pred)

    g = X_test[prot_attrb]

    predictions = {
        'std_model': std_pred[target],
        'fair_model': fair_pred[target]
    }

    start = True

    for name,value in predictions.items():

        DI_value = demographic_parity_ratio(y_true=y_test,y_pred=value,sensitive_features=g)
        mean_diff = demographic_parity_difference(y_true=y_test,y_pred=value,sensitive_features=g)
        eq_odds_diff = equalized_odds_difference(y_true=y_test,y_pred=value,sensitive_features=g)

        if start:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'
        
        with open('./reports/fairness_reports/preprocessing/fairlearn/effnet_gender_report.txt',open_type) as f:
                f.write(f'{name} DI: {round(DI_value,3)}\n')
                f.write(f'{name} mean_diff: {round(mean_diff,3)}\n')
                f.write(f'eq_odds_diff: {round(eq_odds_diff,3)}\n')

def test_fairness(dataset):
    ## funzione che calcola alcune metriche di fairness e cerca di mitigare eventuali discriminazioni presenti nel dataset

    sens_features = ['race']
    dataset = dataset.drop('filename',axis=1)

    dataset= dataset.astype(int)

    corr_remover = CorrelationRemover(sensitive_feature_ids=sens_features,alpha=1.0)

    features_names = dataset.columns.tolist()

    features_names.remove('race')

    fair_dataset = corr_remover.fit_transform(dataset)
    fair_dataset = pd.DataFrame(
        fair_dataset,columns=features_names
    )

    fair_dataset[sens_features] = dataset[sens_features]

    fair_dataset['gender'] = dataset['gender']

    # grafici che mostrano la correlazione prima e dopo
    # sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm')
    # plt.show()

    # sns.heatmap(fair_dataset.corr(),annot=True,cmap='coolwarm')
    # plt.show()

    fair_dataset = fair_dataset.astype(int)
    fair_dataset = fair_dataset.astype(str)

    return fair_dataset

def print_metrics(message,metric,first_message=False):
    if first_message:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/fairness_reports/preprocessing/fairlearn/effnet_gender_report.txt',open_type) as f:
        f.write(f'{message}: {round(metric,3)}\n')

def print_time(time):
    with open('./reports/time_reports/gender/fairlearn/effnet_preprocessing_report.txt','w') as f:
        f.write(f'Elapsed time: {time} seconds.\n')

start = datetime.now()
load_dataset()
end = datetime.now()

elapsed = (end - start).total_seconds()
print_time(elapsed)