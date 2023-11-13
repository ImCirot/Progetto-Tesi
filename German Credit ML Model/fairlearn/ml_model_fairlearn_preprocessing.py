import numpy as np 
from sklearn.metrics import *
import pandas as pd 
from fairlearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fairlearn.preprocessing import CorrelationRemover
import matplotlib.pyplot as plt
from fairlearn.reductions import *
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from codecarbon import track_emissions
import xgboost as xgb
from datetime import datetime
from time import sleep

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

    lr_model_pipeline = pickle.load(open('./output_models/std_models/lr_credit_model.sav','rb'))
    rf_model_pipeline = pickle.load(open('./output_models/std_models/rf_credit_model.sav','rb'))
    svm_model_pipeline = pickle.load(open('./output_models/std_models/svm_credit_model.sav','rb'))
    xgb_model_pipeline = pickle.load(open('./output_models/std_models/xgb_credit_model.sav','rb'))


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
    
    # codice per mostrare una heatmap della correlazione fra attributi prima e dopo la fase di preprocessing
    # sens_and_target_features = ['sex_A91','sex_A92','sex_A93','sex_A94','Duration in month','Credit amount','Installment rate in percentage of disposable income',
    # 'Present residence since','Age in years']
    # sns.heatmap(dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    # plt.title("Standard dataset heatmap")
    # plt.show()

    # sns.heatmap(fair_dataset[sens_and_target_features].corr(),annot=True,cmap='coolwarm')
    # plt.title("Modified dataset heatmap")
    # plt.show()

    X_fair_train, X_fair_test, y_fair_train, y_fair_test = train_test_split(X_fair,y_fair,test_size=0.2,random_state=42)

    print(f'######### Training modelli #########')
    lr_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    rf_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    svm_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())
    xgb_fair_model_pipeline.fit(X_fair_train,y_fair_train.values.ravel())

    # validiamo i risultati prodotti dal modello chiamando una funzione che realizza metriche di valutazione
    print(f'######### Testing modelli #########')
    validate(lr_fair_model_pipeline,'lr',X_fair_test,y_fair_test,True)
    validate(rf_fair_model_pipeline,'rf',X_fair_test,y_fair_test)
    validate(svm_fair_model_pipeline,'svm',X_fair_test,y_fair_test)
    validate(xgb_fair_model_pipeline,'xgb',X_fair_test,y_fair_test)

    print(f'######### Testing Fairness #########')
    lr_std_pred = lr_model_pipeline.predict(X)
    lr_fair_pred = lr_fair_model_pipeline.predict(X_fair)

    rf_std_pred = rf_model_pipeline.predict(X)
    rf_fair_pred = rf_fair_model_pipeline.predict(X_fair)

    svm_std_pred = svm_model_pipeline.predict(X)
    svm_fair_pred = svm_fair_model_pipeline.predict(X_fair)

    xgb_std_pred = xgb_model_pipeline.predict(X)
    xgb_fair_pred = xgb_fair_model_pipeline.predict(X_fair)

    predictions = {
        'lr_std':lr_std_pred,
        'lr_fair': lr_fair_pred,
        'rf_std': rf_std_pred,
        'rf_fair':rf_fair_pred,
        'svm_std': svm_std_pred,
        'svm_fair': svm_fair_pred,
        'xgb_std': xgb_std_pred,
        'xgb_fair': xgb_fair_pred,
    }

    start = True

    for name,prediction in predictions.items():

        sex_DI = demographic_parity_ratio(y_true=y,y_pred=prediction,sensitive_features=sex)

        if start is True:
            open_type = 'w'
            start = False
        else:
            open_type = 'a'

        with open('./reports/fairness_reports/preprocessing/fairlearn/credit_report.txt',open_type) as f:
            f.write(f'{name}_sex DI: {round(sex_DI,3)}\n')

    print(f'######### Salvataggio modelli #########')
    pickle.dump(lr_fair_model_pipeline,open('./output_models/preprocessing_models/lr_fairlearn_credit_model.sav','wb'))
    pickle.dump(rf_fair_model_pipeline,open('./output_models/preprocessing_models/rf_fairlearn_credit_model.sav','wb'))
    pickle.dump(svm_fair_model_pipeline,open('./output_models/preprocessing_models/svm_fairlearn_credit_model.sav','wb'))
    pickle.dump(xgb_fair_model_pipeline,open('./output_models/preprocessing_models/xgb_fairlearn_credit_model.sav','wb'))
    print(f'######### OPERAZIONI TERMINATE CON SUCCESSO #########')

def validate(ml_model,model_type,X_test,y_test,first=False):
    ## funzione utile a calcolare metriche del modello realizzato
    accuracy = ml_model.score(X_test,y_test)

    y_proba = ml_model.predict_proba(X_test)[::,1]

    auc_score = roc_auc_score(y_test,y_proba)

    if first:
        open_type = "w"
    else:
        open_type = "a"
    
    #scriviamo su un file le metriche di valutazione ottenute
    with  open(f"./reports/preprocessing_models/fairlearn/credit_metrics_report.txt",open_type) as f:
        f.write(f"{model_type}\n")
        f.write(f"Accuracy: {round(accuracy,3)}")
        f.write(f'\nROC-AUC score: {round(auc_score,3)}\n')
        f.write('\n')

def load_dataset():
    ## funzione per caricare dataset gia codificato in precedenza
    df = pd.read_csv('./German Credit Dataset/dataset_modificato.csv')

    df.drop('ID', axis=1, inplace=True)

    for i in range(10):
        print(f'########################### {i+1} esecuzione ###########################')
        start = datetime.now()
        training_model(df)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print_time(elapsed,i)
        if(i < 9):
            print('########################### IDLE TIME START ###########################')
            sleep(1)
            print('########################### IDLE TIME FINISH ###########################')
    

def print_time(time,index):
    if index == 0:
        open_type = 'w'
    else:
        open_type = 'a'

    with open('./reports/time_reports/fairlearn/credit_preprocessing_report.txt',open_type) as f:
        f.write(f'{index+1} iter. elapsed time: {time} seconds.\n')

load_dataset()