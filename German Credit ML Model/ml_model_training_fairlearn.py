import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from codecarbon import track_emissions
import xgboost as xgb

@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset):
    ## funzione che addestra il modello sul dataset utilizzando strategia KFold

    # trasformiamo dataset in array per usare indici strategia KFold
    df_array = np.asarray(dataset)

    # evidenziamo le features utili alla predizione
    features = dataset.columns.tolist()

    # rimuoviamo dalla lista features la feature target
    features.remove('Target')

    # evidenziamo gli attributi sensibili del dataset
    sex_features = [
        'sex_A91','sex_A92','sex_A93','sex_A94'
    ]

    # settiamo la nostra X sulle sole variabili di features
    X = dataset[features]

    # settiamo la varibile target con valore 1 e 0 per positivo e negativo, riampiazzando il valore 2 usato come standard per indicare il negativo,
    # operazione necessaria ai fini di utilizzo del toolkit
    dataset['Target'] = dataset['Target'].replace(2,0)

    # settiamo la nostra y sulla variabile da predire
    y = dataset['Target']

    # settiamo un dataframe contenente solamente i valori degli attributi sensibili (utile per utilizzare il framework FairLearn)
    sex = dataset[sex_features]

    # settiamo contatore per ciclo KFold
    i = 0

    # settiamo il numero di ripetizioni uguale a 10, standard per la strategia KFold
    kf = KFold(n_splits=10)

    # Creiamo una pipeline contenente il modello basato su regressione logistica e uno scaler per poter scalare i dati correttamente per poter
    # utilizzare correttamente il modello
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True, class_weight={1:1,0:5}))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )

    # proviamo a rimuovere eventuali correlazioni esistenti fra i dati e le features sensibli
    # utilizzando una classe messa a disposizione dalla libreria FairLearn
    corr_remover = CorrelationRemover(sensitive_feature_ids=sex_features,alpha=1.0)

    # creiamo un nuovo modello da addestrare sul dataset modificato
    lr_fair_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:1,0:5}))
    rf_fair_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:1,0:5}))
    svm_fair_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True,class_weight={1:1,0:5}))
    xgb_fair_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))


    features_names = dataset.columns.tolist()
    for sex_feature in sex_features:
        features_names.remove(sex_feature)
        

    # modifichiamo il set training usando lo strumento di preprocessing fornito dalla libreria FairLearn
    fair_dataset = corr_remover.fit_transform(dataset)
    fair_dataset = pd.DataFrame(
        fair_dataset, columns=features_names
    )

    fair_dataset[sex_features] = dataset[sex_features]
    fair_dataset['Target'] = dataset['Target']
    
    features_list = fair_dataset.columns.tolist()
    features_list.remove('Target')

    X_fair = fair_dataset[features_list]
    y_fair = fair_dataset['Target']
    
    sens_and_target_features = ['sex_A91','sex_A92','sex_A93','sex_A94','Duration in month','Credit amount','Installment rate in percentage of disposable income',
    'Present residence since','Age in years']
    sns.heatmap(dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    plt.title("Standard dataset heatmap")
    plt.show()

    sns.heatmap(fair_dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    plt.title("Modified dataset heatmap")
    plt.show()


    for train_index, test_index in kf.split(df_array):
        i = i + 1

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        sex_train = sex.loc[train_index]

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        sex_test = sex.loc[test_index]

        # fitting del modello sui dati di training per l'iterazione i-esima
        lr_model_pipeline.fit(X_train,y_train.values.ravel())
        rf_model_pipeline.fit(X_train,y_train.values.ravel())
        svm_model_pipeline.fit(X_train,y_train.values.ravel())
        xgb_model_pipeline.fit(X_train,y_train.values.ravel())

        # validiamo i risultati prodotti dal modello all'iterazione i-esima chiamando una funzione che realizza metriche di valutazione
        validate(lr_model_pipeline,"std_models",'lr',i,X_test,y_test)
        validate(rf_model_pipeline,'std_models','rf',i,X_test,y_test)
        validate(svm_model_pipeline,'std_models','svm',i,X_test,y_test)
        validate(xgb_model_pipeline,'std_models','xgb',i,X_test,y_test)

        X_fair_train = X_fair.iloc[train_index]
        y_fair_train = y_fair.iloc[train_index]

        X_fair_test = X_fair.iloc[test_index]
        y_fair_test = y_fair.iloc[test_index]

        # addestriamo il modello
        lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
        rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
        svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
        xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())

        # validiamo i risultati prodotti dal modello all'iterazione i-esima chiamando una funzione che realizza metriche di valutazione
        validate(lr_fair_model_pipeline,'fair_models','lr',i,X_fair_test,y_fair_test)
        validate(rf_fair_model_pipeline,'fair_models','rf',i,X_fair_test,y_fair_test)
        validate(svm_fair_model_pipeline,'fair_models','svm',i,X_fair_test,y_fair_test)
        validate(xgb_fair_model_pipeline,'fair_models','xgb',i,X_fair_test,y_fair_test)

        # modifichiamo i modelli con postop di fairness
        lr_threshold.fit(X_train,y_train,sensitive_features=sex_train)
        rf_threshold.fit(X_train,y_train,sensitive_features=sex_train)
        svm_threshold.fit(X_train,y_train,sensitive_features=sex_train)
        xgb_threshold.fit(X_train,y_train,sensitive_features=sex_train)

        # validiamo i nuovi modelli prodotti
        validate_postop(lr_threshold,'lr',i,X_test,y_test,sex_test)
        validate_postop(rf_threshold,'rf',i,X_test,y_test,sex_test)
        validate_postop(svm_threshold,'svm',i,X_test,y_test,sex_test)
        validate_postop(xgb_threshold,'xgb',i,X_test,y_test,sex_test)

        # linea di codice per plottare il accuracy e selection_rate del modello con operazione di postop
        # plot_threshold_optimizer(lr_threshold)
        # plot_threshold_optimizer(rf_threshold)
        # plot_threshold_optimizer(svm_threshold)
        # plot_threshold_optimizer(xgb_threshold)

    lr_std_pred = lr_model_pipeline.predict(X)
    lr_fair_pred = lr_fair_model_pipeline.predict(X_fair)
    lr_threshold_pred = lr_threshold.predict(X,sensitive_features=sex)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_fair_pred = rf_fair_model_pipeline.predict(X_fair)
    rf_threshold_pred = rf_threshold.predict(X,sensitive_features=sex)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_fair_pred = svm_fair_model_pipeline.predict(X_fair)
    svm_threshold_pred = svm_threshold.predict(X,sensitive_features=sex)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X_fair)
    xgb_threshold_pred = xgb_threshold.predict(X,sensitive_features=sex)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_fair': lr_fair_pred,
        'lr_threshold': lr_threshold_pred,
        'rf_std': rf_std_pred,
        'rf_fair':rf_fair_pred,
        'rf_threshold':rf_threshold_pred,
        'svm_std': svm_std_pred,
        'svm_fair': svm_fair_pred,
        'svm_threshold': svm_threshold_pred,
        'xgb_std': xgb_std_pred,
        'xgb_fair': xgb_fair_pred,
        'xgb_threshold': xgb_threshold_pred
    }

    start = True

    for name,prediction in predictions.items():

        sex_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=sex)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/postprocessing/fairlearn/credit_model_DI.txt',open_type) as f:
            f.write(f'{name}_sex DI: {sex_DI}\n')


    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_fairlearn_credit_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_fairlearn_credit_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_fairlearn_credit_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_fairlearn_credit_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_fairlearn_credit_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_fairlearn_credit_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_fairlearn_credit_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_fairlearn_credit_model.sav','wb'))
    pickle.dump(lr_threshold,open('./output_models/postop_models/threshold_lr_fairlearn_credit_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/postop_models/threshold_rf_fairlearn_credit_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/postop_models/threshold_svm_fairlearn_credit_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/postop_models/threshold_xgb_fairlearn_credit_model.sav','wb'))

def validate(ml_model,model_vers,model_type,index,X_test,y_test):
    ## funzione utile a calcolare metriche del modello realizzato

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
    with open(f"./reports/{model_vers}/fairlearn/credit/{model_type}_credit_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/fairlearn/credit/{model_type}_credit_metrics_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write("Metriche di valutazione:")
        f.write(str(report))
        f.write(f'\nAUC ROC score: {auc_score}\n')
        f.write('\n')

def validate_postop(ml_model,model_type,index,X_test,y_test,g_test):
    pred = ml_model.predict(X_test,sensitive_features=g_test)

    matrix = confusion_matrix(y_test,pred)

    report = classification_report(y_test,pred)

    if index == 1:
        open_type = 'w'
    else:
        open_type = 'a'

    # scriviamo su un file la matrice di confusione ottenuta
    with open(f'./reports/postop_models/fairlearn/credit/{model_type}_credit_matrix_report.txt', open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write(f'Matrice di confusione:\n')
        f.write(str(matrix))
        f.write('\n\n')

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/postop_models/fairlearn/credit/{model_type}_credit_metrics_report.txt',open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write('Metriche di valutazione:')
        f.write(str(report))
        f.write('\n')

def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./German Credit Dataset/dataset_modificato.csv')

    df.drop('ID', axis=1, inplace=True)

    training_model(df)


load_dataset()