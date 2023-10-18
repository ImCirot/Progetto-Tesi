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
def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./Student Dataset/dataset.csv')

    df.drop('ID', axis=1, inplace=True)

    training_model(df)

def training_model(dataset):

    df_array = np.asarray(dataset)

    features = dataset.columns.tolist()

    features.remove("Target")

    protected_features = [
        'Gender','Educational special needs'
    ]

    X = dataset[features]

    y = dataset['Target']

    g = dataset[protected_features]

    i = 0

    kf = KFold(n_splits=10)

    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    lr_threshold = ThresholdOptimizer(
        estimator=lr_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False,
        objective='accuracy_score'
    )

    rf_threshold = ThresholdOptimizer(
        estimator=rf_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False,
        objective='accuracy_score'
    )

    svm_threshold = ThresholdOptimizer(
        estimator=svm_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False,
        objective='accuracy_score'
    )

    xgb_threshold = ThresholdOptimizer(
        estimator=xgb_model_pipeline,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=False,
        objective='accuracy_score'
    )

    # creiamo un nuovo modello da addestrare sul dataset modificato
    lr_fair_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    rf_fair_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_fair_model_pipeline = make_pipeline(StandardScaler(),SVC(probability=True))
    xgb_fair_model_pipeline = make_pipeline(StandardScaler(),xgb.XGBClassifier(objective='binary:logistic', random_state=42))

    fair_dataset = fairness_preprocess_op(dataset,protected_features)

    features_list = fair_dataset.columns.tolist()
    features_list.remove('Target')

    X_fair = fair_dataset[features_list]
    y_fair = fair_dataset['Target']
    
    for train_index, test_index in kf.split(df_array):
        i = i + 1

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # estraiamo parte di training dal dataset per il ciclo i-esimo
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

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
        
        g_test = g.iloc[test_index]
        g_train = g.iloc[train_index]

        lr_threshold.fit(X_train,y_train,sensitive_features=g_train)
        rf_threshold.fit(X_train,y_train,sensitive_features=g_train)
        svm_threshold.fit(X_train,y_train,sensitive_features=g_train)
        xgb_threshold.fit(X_train,y_train,sensitive_features=g_train)

        # validiamo i nuovi modelli prodotti
        validate_postop(lr_threshold,'lr',i,X_test,y_test,g_test)
        validate_postop(rf_threshold,'rf',i,X_test,y_test,g_test)
        validate_postop(svm_threshold,'svm',i,X_test,y_test,g_test)
        validate_postop(xgb_threshold,'xgb',i,X_test,y_test,g_test)

    lr_std_pred = lr_model_pipeline.predict(X)
    lr_fair_pred = lr_fair_model_pipeline.predict(X_fair)
    lr_threshold_pred = lr_threshold.predict(X,sensitive_features=g)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_fair_pred = rf_fair_model_pipeline.predict(X_fair)
    rf_threshold_pred = rf_threshold.predict(X,sensitive_features=g)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_fair_pred = svm_fair_model_pipeline.predict(X_fair)
    svm_threshold_pred = svm_threshold.predict(X,sensitive_features=g)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X_fair)
    xgb_threshold_pred = xgb_threshold.predict(X,sensitive_features=g)

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

        DI_score = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=g)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/student_model_DI.txt',open_type) as f:
            f.write(f'{name} DI: {DI_score}\n')
    
    # linea di codice per plottare il accuracy e selection_rate del modello con operazione di postop
    # plot_threshold_optimizer(lr_threshold)
    # plot_threshold_optimizer(rf_threshold)
    # plot_threshold_optimizer(svm_threshold)
    # plot_threshold_optimizer(xgb_threshold)

    pickle.dump(lr_model_pipeline,open('./output_models/std_models/lr_fairlearn_student_model.sav','wb'))
    pickle.dump(lr_fair_model_pipeline,open('./output_models/fair_models/lr_fairlearn_student_model.sav','wb'))
    pickle.dump(rf_model_pipeline,open('./output_models/std_models/rf_fairlearn_student_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/fair_models/rf_fairlearn_student_model.sav','wb'))
    pickle.dump(svm_model_pipeline,open('./output_models/std_models/svm_fairlearn_student_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/fair_models/svm_fairlearn_student_model.sav','wb'))
    pickle.dump(xgb_model_pipeline,open('./output_models/std_models/xgb_fairlearn_student_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/fair_models/xgb_fairlearn_student_model.sav','wb'))
    pickle.dump(lr_threshold,open('./output_models/postop_models/threshold_lr_fairlearn_student_model.sav','wb'))
    pickle.dump(rf_threshold,open('./output_models/postop_models/threshold_rf_fairlearn_student_model.sav','wb'))
    pickle.dump(svm_threshold,open('./output_models/postop_models/threshold_svm_fairlearn_student_model.sav','wb'))
    pickle.dump(xgb_threshold,open('./output_models/postop_models/threshold_xgb_fairlearn_student_model.sav','wb'))

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
    with open(f"./reports/{model_vers}/fairlearn/student/{model_type}_student_matrix_report.txt",open_type) as f:
        f.write(f"{index} iterazione:\n")
        f.write(f"Matrice di confusione:\n")
        f.write(str(matrix))
        f.write('\n\n')
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/{model_vers}/fairlearn/student/{model_type}_student_metrics_report.txt",open_type) as f:
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
    with open(f'./reports/postop_models/{model_type}_student_matrix_report.txt', open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write(f'Matrice di confusione:\n')
        f.write(str(matrix))
        f.write('\n\n')

    # scriviamo su un file le metriche di valutazione ottenute
    with open(f'./reports/postop_models/{model_type}_student_metrics_report.txt',open_type) as f:
        f.write(f'{index} iterazione:\n')
        f.write('Metriche di valutazione:')
        f.write(str(report))
        f.write('\n')

def fairness_preprocess_op(dataset, protected_features_names):
    ## funzione che utilizza classe offerta da fairlearn in grado di mitigare la correlazione fra gli attributi sensibili e non del dataset

    # lista contenente nomi colonne del dataset escluse le feature sensibili
    features_names = dataset.columns.tolist()
    for feature in protected_features_names:
        features_names.remove(feature)

    # creiamo un oggetto fornito dalla libreria Fairlean in grado di rimuovere correlazione fra le feature del dataset e le feature sensibili
    corr_remover = CorrelationRemover(sensitive_feature_ids=protected_features_names,alpha=1.0)
    fair_dataset = corr_remover.fit_transform(dataset)

    # ricostruiamo il dataframe inserendo le feature appena modificate
    fair_dataset = pd.DataFrame(
        fair_dataset, columns=features_names
    )

    # inseriamo nel nuovo dataframe le variabili sensibili rimosse in precedenza e la variabile target
    fair_dataset[protected_features_names] = dataset[protected_features_names]
    fair_dataset['Target'] = dataset['Target']

    # stampiamo heatmap che confrontano il grado di correlazione fra gli attributi sensibili e alcuni attributi del dataset
    show_correlation_heatmap(dataset=dataset,title='original dataset')
    show_correlation_heatmap(dataset=fair_dataset,title='modified dataset')

    return fair_dataset

def show_correlation_heatmap(dataset,title):
    ## funzione che genera heatmap sul dataset fornito usando attributi sensibili e non per mostrare il grado di correlazione presente
    
    # creiamo una heatmap del dataset che mostra la correlazione fra gli attributi dichiarati in precedenza
    sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm',fmt='.2f',)
    plt.title(title)
    plt.show()

load_dataset()
