import mne
import numpy as np
import scipy.io as sio
import glob
import os

from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical # type: ignore
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
