import mne
import numpy as np
import scipy.io as sio
import glob
import os

from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
#from preprocess_HGD import load_HGD_data

def load_data_LOSO(data_path,subject,dataset):
    X_train, y_train = [], []
    for sub in range (0,9):
        #path = data_path+'s' + str(sub+1) + '/'
        path = data_path+'/'
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'BCI2b'):
            X1, y1 = load_BCI2b_data(path, sub+1, True)
            X2, y2 = load_BCI2b_data(path, sub+1, False)
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif len(X_train) == 0:  
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


def load_BCI2b_data(data_path, subject, training, all_trials=True):
    """
    Carica e combina i dati EEG del dataset BCI Competition IV-2b da più sessioni.

    Parametri
    ----------
    data_path : str
        Percorso alla cartella contenente i file .gdf.
    subject : int
        Numero del soggetto [1, ..., 9].
    training : bool
        Se True, carica le sessioni di training (T).
        Se False, carica le sessioni di test (E).
    all_trials : bool
        Se True, carica tutti i trials.
        Se False, esclude i trials con artefatti.

    Ritorna
    -------
    data_return : np.ndarray
        Array con i dati EEG (n_trials, n_channels, window_length).
    class_return : np.ndarray
        Array con le etichette dei trials.
    """

    # Identifica i file in base al soggetto e al tipo (T = Training, E = Test)
    file_suffix = 'T' if training else 'E'
    subject_files = sorted(glob.glob(os.path.join(data_path, f"B{subject:02d}??{file_suffix}.gdf")))

    if not subject_files:
        raise FileNotFoundError(f"Nessun file trovato per il soggetto {subject} e sessione {file_suffix} in {data_path}")

    all_data = []
    all_labels = []

    # Frequenza di campionamento e finestra temporale
    fs = 250  # Hz
    t1, t2 = int(0.5 * fs), int(3.5 * fs)  # Intervallo temporale
    window_length = t2 - t1

    # Canali EEG da selezionare
    channels = ['C3', 'Cz', 'C4']

    for gdf_file in subject_files:
        #print(f"Caricando file GDF: {gdf_file}")

        # Caricamento dati EEG
        raw = mne.io.read_raw_gdf(gdf_file, preload=True)
        raw.filter(l_freq=0.5, h_freq=100)  # Filtraggio passa banda
        #print(f"Canali disponibili nel file: {raw.ch_names}")

        # Normalizza i nomi dei canali rimuovendo il prefisso 'EEG:'
        normalized_ch_names = [ch.split(':')[-1] for ch in raw.ch_names]

        # Verifica se i canali richiesti sono effettivamente presenti
        available_channels = [ch for ch in channels if ch in normalized_ch_names]
        if not available_channels:
            print(f"Nessuno dei canali richiesti ({channels}) è disponibile nel file {gdf_file}. Salto il file.")
            continue

        # Seleziona i canali disponibili
        raw.pick_channels([f"EEG:{ch}" for ch in available_channels])
        #print(f"Canali selezionati: {raw.ch_names}")

        # Estrai gli eventi dal file GDF
        events, event_id = mne.events_from_annotations(raw)
        #print(f"Eventi trovati: {event_id}")

        # Mappa gli eventi alle classi previste
        class_mapping = {1: 0, 2: 1}  # Mappa evento 1 -> classe 0, evento 2 -> classe 1
        trial_events = [e for e in events if e[2] in class_mapping]
        trial_labels = [class_mapping[e[2]] for e in trial_events]  # Etichette dei trial
        trial_markers = [e[0] for e in trial_events]  # Inizio dei trial

        #print(f"Numero totale di trial: {len(trial_labels)}")

        for i, start in enumerate(trial_markers):
            if not all_trials and 'artefact' in event_id and trial_labels[i] == event_id['artefact']:
                print(f"Trial {i} scartato per artefatti")
                continue

            try:
                # Verifica che gli indici siano validi
                if start + t1 < 0 or start + t2 > raw.n_times:
                    print(f"Indici non validi per il trial {i}: start={start + t1}, stop={start + t2}")
                    continue

                trial_data = raw.get_data(start=start + t1, stop=start + t2)  # (n_channels, window_length)
                all_data.append(trial_data)
                all_labels.append(trial_labels[i])
            except ValueError as e:
                print(f"Errore durante l'estrazione dei dati per il trial {i}: {e}")
                continue

    # Conversione in array numpy
    if all_data:
        data_return = np.array(all_data)  # (n_trials, n_channels, window_length)
        class_return = np.array(all_labels)  # Etichette già mappate a [0, 1]
        assert np.all(class_return >= 0) and np.all(class_return < 2), "Le etichette non sono nel range [0, 1]"
        #print(f"Dati EEG caricati: {data_return.shape}")
        #print(f"Etichette caricate: {class_return.shape}")
    else:
        print("Nessun dato valido trovato.")
        data_return = np.empty((0, len(channels), window_length))
        class_return = np.empty((0,))

    return data_return, class_return

"""Nuova Funzione BCI2b load_data per evitare un k-score basso
def load_BCI2b_data(data_path, subject, training, all_trials=True):
    
    #Versione migliorata con:
    #- Migliore gestione degli artefatti
    #- Filtraggio più appropriato
    #- Verifica del bilanciamento delle classi
    #- Gestione più robusta degli eventi
    
    file_suffix = 'T' if training else 'E'
    subject_files = sorted(glob.glob(os.path.join(data_path, f"B{subject:02d}??{file_suffix}.gdf")))
    
    if not subject_files:
        raise FileNotFoundError(f"Nessun file trovato per il soggetto {subject} e sessione {file_suffix} in {data_path}")

    all_data = []
    all_labels = []

    fs = 250  # Hz
    t1, t2 = int(0.5 * fs), int(3.5 * fs)  # 0.5-3.5 secondi dopo l'evento
    window_length = t2 - t1

    # Canali EEG - aggiungo anche Pz per migliorare la copertura
    channels = ['C3', 'Cz', 'C4', 'Pz']

    for gdf_file in subject_files:
        try:
            # Caricamento con correzione automatica dei nomi dei canali
            raw = mne.io.read_raw_gdf(gdf_file, preload=True)
            
            # Filtraggio migliorato per MI (8-30 Hz)
            raw.filter(8, 30, method='iir', iir_params=dict(order=5, ftype='butter'))
            
            # Notch filter per rimuovere rumore di linea (50Hz in Europa)
            raw.notch_filter(50)
            
            # Normalizza i nomi dei canali
            normalized_ch_names = [ch.split(':')[-1] if ':' in ch else ch for ch in raw.ch_names]
            raw.rename_channels(dict(zip(raw.ch_names, normalized_ch_names)))
            
            # Seleziona solo i canali EEG disponibili
            available_channels = [ch for ch in channels if ch in raw.ch_names]
            if len(available_channels) < 2:  # Almeno 2 canali
                print(f"Canali insufficienti nel file {gdf_file}. Disponibili: {raw.ch_names}")
                continue
                
            raw.pick_channels(available_channels)
            
            # Estrai eventi con gestione più robusta
            events, event_id = mne.events_from_annotations(raw)
            
            # Mappa eventi alle classi con controllo aggiuntivo
            class_mapping = {'769': 0, '770': 1, '771': 2, '772': 3}  # Standard per BCI2b
            trial_events = []
            trial_labels = []
            
            for e in events:
                if str(e[2]) in class_mapping:
                    trial_events.append(e)
                    trial_labels.append(class_mapping[str(e[2])])
            
            if not trial_events:
                print(f"Nessun evento valido trovato in {gdf_file}")
                continue
                
            # Estrazione dei trial con controllo degli artefatti
            for i, (start, _, event_code) in enumerate(trial_events):
                event_str = str(event_code)
                
                # Salta trial con artefatti se richiesto
                if not all_trials and ('artefact' in event_id or 'artif' in event_id):
                    print("Trial con artefatti esclusi")
                    continue
                    
                try:
                    # Estrai dati con controllo dei limiti
                    if start + t1 >= 0 and start + t2 <= raw.n_times:
                        trial_data = raw.get_data(start=start + t1, stop=start + t2)
                        
                        # Controllo qualità: scarta trial con valori NaN o estremi
                        if np.isnan(trial_data).any() or np.max(np.abs(trial_data)) > 1e6:
                            print(f"Trial {i} scartato per dati non validi")
                            continue
                            
                        all_data.append(trial_data)
                        all_labels.append(class_mapping[event_str])
                except Exception as e:
                    print(f"Errore durante l'estrazione del trial {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {gdf_file}: {e}")
            continue
            
    # Conversione e controllo finale
    if all_data:
        data_return = np.array(all_data)
        class_return = np.array(all_labels)
        
        # Verifica bilanciamento classi
        unique, counts = np.unique(class_return, return_counts=True)
        print(f"Distribuzione classi: {dict(zip(unique, counts))}")
        
        # Rimozione eventuali NaN residui
        mask = ~np.isnan(data_return).any(axis=(1,2))
        data_return = data_return[mask]
        class_return = class_return[mask]
    else:
        data_return = np.empty((0, len(channels), window_length))
        class_return = np.empty((0,))
        
    return data_return, class_return
"""

def load_BCI2a_data(data_path, subject, training, all_trials=True):
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return

def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

"""
def standardize_data(X_train, X_test, channels):
    
    #Versione migliorata con:
    #- Standardizzazione più robusta mantenendo la struttura spaziotemporale
    #- Gestione di eventuali NaN
    #- Opzione per standardizzazione globale
    
    # Conserva la forma originale
    original_shape_train = X_train.shape
    original_shape_test = X_test.shape
    
    # Reshape per standardizzazione (considera tutti i punti temporali insieme)
    X_train_reshaped = X_train.reshape(-1, channels, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, channels, X_test.shape[-1])
    
    # Standardizzazione per ogni canale
    for j in range(channels):
        scaler = StandardScaler()
        
        # Estrai tutti i trial per il canale j
        train_channel = X_train_reshaped[:, j, :]
        test_channel = X_test_reshaped[:, j, :]
        
        # Rimuovi eventuali NaN
        if np.isnan(train_channel).any():
            train_channel = np.nan_to_num(train_channel, nan=np.nanmean(train_channel))
        
        # Fit e transform
        scaler.fit(train_channel)
        X_train_reshaped[:, j, :] = scaler.transform(train_channel)
        X_test_reshaped[:, j, :] = scaler.transform(test_channel)
    
    # Ripristina la forma originale
    X_train = X_train_reshaped.reshape(original_shape_train)
    X_test = X_test_reshaped.reshape(original_shape_test)
    
    return X_train, X_test
"""

def get_data(path,subject,dataset='BCI2b',class_labels='all',LOSO=False,isStandard=True,isShuffle=True):
    # Load and split the dataset into training and testing 
    if LOSO:
        """ Loading and Dividing of the dataset based on the 
        'Leave One Subject Out' (LOSO) evaluation approach. """ 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        """ Loading and Dividing of the data set based on the subject-specific 
        (subject-dependent) approach.
        In this approach, we used the same training and testing data as the original
        competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1 
        for training, and 288 x 9 trials in session 2 for testing.  
        """
        if (dataset == 'BCI2a'):
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        elif(dataset == 'BCI2b'):
            #Path per i miei dati B0x0xT.gdf
            path = path + '/'
            X_train, y_train = load_BCI2b_data(path, subject+1, True)
            X_test, y_test = load_BCI2b_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train,num_classes=2)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test,num_classes=2)
    
    print(f"y_train_onehot shape: {y_train_onehot.shape}")
    print(f"y_test_onehot shape: {y_test_onehot.shape}")    
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

"""
def get_data(path, subject, dataset='BCI2b', class_labels='all', LOSO=False, 
             isStandard=True, isShuffle=True, balance_classes=True):
    
    #Versione migliorata con:
    #- Opzione per bilanciamento delle classi
    #- Migliore gestione degli shape
    #- Controllo della distribuzione delle classi
    #- Aggiunta di logging informativo
    
    # Caricamento dati originale
    if LOSO:
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        if dataset == 'BCI2a':
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        elif dataset == 'BCI2b':
            path = path + '/'
            X_train, y_train = load_BCI2b_data(path, subject+1, True)
            X_test, y_test = load_BCI2b_data(path, subject+1, False)
        else:
            raise ValueError(f"Dataset '{dataset}' non supportato")

    # Analisi distribuzione classi
    print("\nDistribuzione classi prima del bilanciamento:")
    print(f"Training - Classe 0: {sum(y_train == 0)}, Classe 1: {sum(y_train == 1)}")
    print(f"Test - Classe 0: {sum(y_test == 0)}, Classe 1: {sum(y_test == 1)}")

    # Bilanciamento delle classi (solo training set)
    if balance_classes:
        from imblearn.over_sampling import SMOTE
        
        # Reshape per SMOTE (flatten temporale)
        N_tr, N_ch, T = X_train.shape
        X_train_flat = X_train.reshape(N_tr, -1)  # (trials, channels*time_points)
        
        # Applica SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
        
        # Ripristina la forma originale
        X_train = X_train_balanced.reshape(-1, N_ch, T)
        y_train = y_train_balanced
        
        print("\nDopo bilanciamento:")
        print(f"Training - Classe 0: {sum(y_train == 0)}, Classe 1: {sum(y_train == 1)}")

    # Shuffle
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Reshape per modello CNN (aggiungi dimensione MI-tasks)
    X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])

    # One-hot encoding
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    # Standardizzazione
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, X_train.shape[2])  # Numero canali

    # Verifica finale
    print("\nShape finali:")
    print(f"X_train: {X_train.shape}, y_train: {y_train_onehot.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test_onehot.shape}")
    
    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
    
"""