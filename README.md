# Progetto-Tesi
Sviluppo di diversi modelli di ML per testarne fairness tramite le librerie "FairLearn" e "AIFairness360", la qualità del modello generato e la sostenibilità del prodotto.

<h1>Dipendenze</h1>

Per poter utilizzare correttamente questo progetto è necessario avere installato **Python** (vers. >=3.11.5).<br>
È consigliato, inoltre, creare un ambiente virtuale python sfruttando o **anaconda** o **venv**.<br>
Qui i riferimenti su come creare un Virtual Enviroment usando [venv](https://docs.python.org/3/library/venv.html) o [anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

Sono necessari i seguenti PythonPackages:
```
- Numpy
- Pandas
- Seaborn
- Matplotlib
- ScikitLearn
- AIF360
- TensorFlow
- CodeCarbon
- FairLearn
- XGBoost
```
È possibile installare le ultime versioni di questi pacchetti tramite il comando:
```
pip install numpy pandas seaborn matplotlib sklearn aif360 codecarbon fairlearn xgboost
```
Per l'installazione di TensorFlow, riferirsi alla guida ufficiale [qui](https://www.tensorflow.org/install).

<h1>Organizzazione e struttura del progetto</h1>
Il progetto è organizzato in diverse directory. Di seguito la struttura del progetto:

- **Adult Dataset**: contiene l'Adult dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/2/adult).
- **German Credit Dataset**: contiene il "German Credit" dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
- **Bank Marketing Dataset**: contiene il "Bank Marketing" dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- **Student Dataset**: contiene lo "Student dropout and accademic success" dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
- **UTKFace Dataset**: contiene il dataset **UTKFace** disponibile al seguente [link](https://susanqq.github.io/UTKFace/).<br>
  **IMPORTANTE: cartella non presente nella repository per via delle dimensioni è necessario scaricare il dataset dal link fornito e creare questa cartella con il contenuto del dataset**.
- **Formatting Data**: contiene script utili per la manipolazione e preparazione dei dataset.
- **Adult ML Model**: contiene gli script python per realizzare modelli sulla base dell'Adult dataset, sfruttando ognuno una delle librerie di fairness citate.
- **German Credit ML Model**: contiene script python per realizzare modelli sulla base del German Credit dataset, sfruttando ognuno una delle librerie di fairness citate.
- **Bank Marketing ML Model**: contiene script python per realizzare modelli sulla base del Bank Marketing dataset, sfruttando ognuno una delle librerie di fairness citate.
- **Student ML Model**: contiene script python per realizzare modelli sulla base dello Student dropout and accademic success dataset, sfruttando ognuno una delle librerie di fairness citate.
- **Gender Recognition Model**: contiene script python per realizzare modelli di gender recognition sfruttando il dataset UTKFace.
- **reports**: contiene tutti i report generati dagli script durante la fase di training e testing.
  - **fair_models**: contiene metriche di valutazione (matrice di confusione/accuracy/f1 score...) per i modelli addestrati su dataset che hanno subito operazioni di fairness.
    - **aif360**: contiene metriche di valutazione sui modelli addestrati su dataset su cui è stata utilizzata la libreria aif360 per valutare fairness e mitigare possibili disciminazioni.
      - **adult**: contiene metriche di valutazione sui modelli addestrati su dataset Adult.
      - **bank**: contiene metriche di valutazione sui modelli addestrati su dataset Bank Marketing.
      - **credit**: contiene metriche di valutazione sui modelli addestrati su dataset German Credit.
      - **student**: contiene metriche di valutazione sui modelli addestrati su dataset Student dropout and accademic success.
    - **fairlearn**: contiene metriche di valutazione sui modelli addestrati su dataset su cui è stata utilizzata la libreria fairlearn per valutare fairness e mitigare possibili disciminazioni.
      - **adult**: contiene metriche di valutazione sui modelli addestrati su dataset Adult.
      - **bank**: contiene metriche di valutazione sui modelli addestrati su dataset Bank Marketing.
      - **credit**: contiene metriche di valutazione sui modelli addestrati su dataset German Credit.
      - **student**: contiene metriche di valutazione sui modelli addestrati su dataset Student dropout and accademic success.
  - **std_models**: stessa struttura della directory precedente. Contiene metriche di valutazione sui modelli addestrati sui dataset originali.
  - **fairness_reports**: contiene reports di metriche di fairness dei diversi datasets.
  - **postop_models**: contiene metriche di valutazione dei modelli ottenuti tramite operazioni di postprocessing sui modelli standard.
- **output_models**: contiene tutti i modelli generati salvati.<br>
  **IMPORTANTE: cartella non presente nella repository per via delle dimensioni, è necessario crearla per evitare errori di directory mancante**
- **emission.csv**: file generato dalla libreria CodeCarbon che contiene le valutazioni energetiche e di consumo di diverse esecuzioni di codice.
