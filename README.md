# Progetto-Tesi
Sviluppo di diversi modelli di ML per testarne fairness tramite le librerie "FairLearn" e "AIFairness360", la qualità del modello generato e la sostenibilità del prodotto.

<h1>Organizzazione e struttura del progetto</h1>
Il progetto è organizzato in diverse directory. Di seguito la struttura del progetto:

- **Adult Dataset**: contiene l'Adult dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/2/adult).
- **German Credit Dataset**: contiene il "German Credit" dataset originale e le sue versioni modificate per poter usufruire degli strumenti di fairness. Il dataset originale è disponibile [qui](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
- **Formatting Data**: contiene script utili per la manipolazione e preparazione dei dataset.
- **Adult ML Model**: contiene gli script python per realizzare modelli sulla base dell'Adult dataset, sfruttando ognuno una delle librerie di fairness citate.
- **German Credit ML Model**: contiene script python per realizzare modelli sulla base del German Credit dataset, sfruttando ognuno una delle librerie di fairness citate.
- **reports**: contiene tutti i report generati dagli script durante la fase di training e testing.
  - **fair_models**: contiene metriche di valutazione (matrice di confusione/accuracy/f1 score...) per i modelli addestrati su dataset che hanno subito operazioni di fairness.
    - **aif360**: contiene metriche di valutazione sui modelli addestrati su dataset su cui è stata utilizzata la libreria aif360 per valutare fairness e mitigare possibili disciminazioni.
    - **fairlearn**: contiene metriche di valutazione sui modelli addestrati su dataset su cui è stata utilizzata la libreria fairlearn per valutare fairness e mitigare possibili disciminazioni.
  - **std_models**: stessa struttura della directory precedente. Contiene metriche di valutazione sui modelli addestrati sui dataset originali.
  - **fairness_reports**: contiene reports di metriche di fairness dei modelli.
  - **emission.csv**: file generato dalla libreria CodeCarbon che contiene le valutazioni energetiche e di consumo di diverse esecuzioni di codice.
