""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%
def load_data_LOSO (data_path, subject, dataset): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model.
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)
        # elif (dataset == 'HGD'):
        #     X1, y1 = load_HGD_data(path, sub+1, True)
        #     X2, y2 = load_HGD_data(path, sub+1, False)
        
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


"""
def load_data_LOSO(data_path, subject, dataset):
    X_train_list, y_train_list = [], []
    X_test, y_test = None, None

    for sub in range(14):
        path = data_path + '/'
        X1, y1 = load_HGD_data(path, sub+1, True)
        X2, y2 = load_HGD_data(path, sub+1, False)

        if sub == subject:
            # SOLO sessione 2 del left-out
            X_test, y_test = X2, y2
        else:
            # SOLO sessione 1 degli altri
            X_train_list.append(X1)
            y_train_list.append(y1)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, y_train, X_test, y_test
"""

#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):
    """ Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
    """
    
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

#%%
import os
import glob
import numpy as np
import mne
from collections import Counter
from sklearn.utils import resample

def balance_by_SMOTE_or_duplication(data, labels, method="auto", k_base=5):
    from collections import Counter
    from sklearn.utils import resample
    from imblearn.over_sampling import SMOTE

    counter = Counter(labels)
    print(f"[DEBUG] Conteggio classi prima: {counter}")
    
    min_class = min(counter.values())
    k_neighbors = min(k_base, min_class - 1)

    if method == "SMOTE" or (method == "auto" and k_neighbors >= 1):
        print(f"[INFO] Applico SMOTE con k_neighbors={k_neighbors}")
        X = data.reshape(len(data), -1)
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X, labels)
        print(f"[DEBUG] Conteggio classi dopo SMOTE: {Counter(y_res)}")
        return X_res.reshape(-1, data.shape[1], data.shape[2]), y_res

    elif method == "duplicate" or (method == "auto" and k_neighbors < 1):
        print("[INFO] Uso duplicazione per bilanciare le classi")
        max_count = max(counter.values())
        data_balanced, labels_balanced = [], []

        for label in np.unique(labels):
            class_data = data[labels == label]
            class_labels = labels[labels == label]
            class_data_res, class_labels_res = resample(
                class_data, class_labels,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            data_balanced.append(class_data_res)
            labels_balanced.append(class_labels_res)

        final_labels = np.concatenate(labels_balanced)
        print(f"[DEBUG] Conteggio classi dopo duplicazione: {Counter(final_labels)}")
        return np.concatenate(data_balanced), final_labels

    else:
        raise ValueError(f"Metodo di bilanciamento non valido: {method}")
    
def add_jitter(X, sigma=0.02):
    """
    Aggiunge rumore gaussiano ai dati EEG (data augmentation).

    Parametri
    ----------
    X : np.ndarray
        Dati EEG di forma (n_channels, window_samples)
    sigma : float
        Deviazione standard del rumore

    Ritorna
    -------
    X_noisy : np.ndarray
        Dati con rumore aggiunto
    """
    return X + sigma * np.random.randn(*X.shape)


def load_BCI2b_data(data_path, subject, training=True,
                            window_sec=3.0, step_sec=0.5,
                            augment=False, balance=False, verbose=False):
    import mne
    import glob
    import numpy as np
    import os
    from collections import Counter

    fs = 250
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    target_channels = ['C3', 'Cz', 'C4']

    file_suffix = 'T' if training else 'E'
    subject_files = sorted(glob.glob(os.path.join(data_path, f"B{subject:02d}??{file_suffix}.gdf")))

    if not subject_files:
        raise FileNotFoundError(f"Nessun file trovato per il soggetto {subject} ({file_suffix})")

    segments, labels = [], []

    for gdf_file in subject_files:
        raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
        raw.filter(0.5, 45, fir_design='firwin', verbose=False)
        raw.notch_filter(50, verbose=False)

        sel_channels = [ch for ch in raw.info["ch_names"] if any(t in ch for t in target_channels)]
        raw.pick_channels(sel_channels)

        events, _ = mne.events_from_annotations(raw)
        label_map = {1: 0, 2: 1}
        events = [e for e in events if e[2] in label_map]

        for e in events:
            trial_start = e[0] + int(0.5 * fs)
            trial_end = trial_start + int(4.0 * fs)

            for t in range(trial_start, trial_end - window_samples + 1, step_samples):
                if t + window_samples > raw.n_times:
                    continue
                seg = raw.get_data(start=t, stop=t + window_samples)

                # Standardizzazione per canale
                seg = (seg - np.mean(seg, axis=1, keepdims=True)) / \
                      (np.std(seg, axis=1, keepdims=True) + 1e-6)

                # Augment EEG-aware
                if training and augment:
                    if np.random.rand() < 0.3:
                        seg += 0.005 * np.random.randn(*seg.shape)  # jitter lieve
                    if np.random.rand() < 0.2:
                        shift = np.random.randint(5, 15)
                        seg = np.roll(seg, shift=shift, axis=1)  # time-shifting
                    if np.random.rand() < 0.1:
                        seg = seg * (1 + 0.01 * np.random.randn(*seg.shape))  # lievi distorsioni

                segments.append(seg)
                labels.append(label_map[e[2]])

    if not segments:
        raise ValueError(f"[ERRORE] Nessun segmento trovato per il soggetto {subject}")

    X = np.stack(segments)
    y = np.array(labels)

    if balance and training:
        counts = np.bincount(y)
        if len(counts) == 2 and counts[0] != counts[1]:
            maj_class = 0 if counts[0] > counts[1] else 1
            diff = abs(counts[0] - counts[1])
            idx_min = np.where(y != maj_class)[0]
            X_extra = X[idx_min[:diff]]
            y_extra = y[idx_min[:diff]]
            X = np.concatenate([X, X_extra], axis=0)
            y = np.concatenate([y, y_extra], axis=0)
            if verbose:
                print(f"[INFO] Bilanciamento duplicando classe minoritaria. Classi: {Counter(y)}")

    if verbose:
        print(f"[INFO] Soggetto {subject} - Segmenti totali: {X.shape[0]}, Finestra: {window_sec}s, Step: {step_sec}s")

    return X, y

#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

#%%
def get_data(path, subject, dataset = 'BCI2b', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True):
    
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
        elif (dataset == 'BCI2b'):
            path = path + '/'
            X_train, y_train = load_BCI2b_data(path, subject+1, True,augment=True, balance=True)
            X_test, y_test = load_BCI2b_data(path, subject+1, False)
            
        # elif (dataset == 'HGD'):
        #     X_train, y_train = load_HGD_data(path, subject+1, True)
        #     X_test, y_test = load_HGD_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)    
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
