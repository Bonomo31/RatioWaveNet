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

#%%
# We need the following to load and preprocess the High Gamma Dataset
import mne
import numpy as np
import logging
from collections import OrderedDict

#Importo le funzioni dalla cartella libreria_braindecode locale
from libreria_braindecode.bbci import BBCIDataset
from libreria_braindecode.trial_segment import create_signal_target_from_raw_mne
from libreria_braindecode.signalproc import resample_cnt

#from braindecode.datasets.bbci import BBCIDataset
#from braindecode.datautil.trial_segment import \
#    create_signal_target_from_raw_mne
#from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signalproc import highpass_cnt

#%%
def load_HGD_data(data_path, subject, training, low_cut_hz =0, debug = False):
    """ Loading training/testing data for the High Gamma Dataset (HGD)
    for a specific subject.
    
    Please note that  HGD is for "executed movements" NOT "motor imagery"  
    
    This code is taken from https://github.com/robintibor/high-gamma-dataset 
    You can download the HGD using the following link: 
        https://gin.g-node.org/robintibor/high-gamma-dataset/src/master/data
    The Braindecode library is required to load and processs the HGD dataset.
   
        Parameters
        ----------
        data_path: string
            dataset path
        subject: int
            number of subject in [1, .. ,14]
        training: bool
            if True, load training data
            if False, load testing data
        debug: bool
            if True, 
            if False, 
    """

    log = logging.getLogger(__name__)
    log.setLevel('DEBUG')

    if training:  filename = (data_path + 'train/{}.mat'.format(subject))
    else:         filename = (data_path + 'test/{}.mat'.format(subject))

    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)
    
    log.info("Loading data...")
    cnt = loader.load()
    
    # Salviamo il canale di stimolazione prima di eliminarlo
    stim_channel = cnt.copy().pick_channels(['STI 014'])


    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [2]), ('Left Hand', [4],),
                              ('Rest', [6]), ('Feet', [8])])
    clean_ival = [0, 3000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)



    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))
    


    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    # Ora selezioniamo solo i canali EEG
    cnt = cnt.pick_channels(C_sensors)
    
    # Riaggiungiamo il canale di stimolazione
    cnt = cnt.add_channels([stim_channel])

    log.info("Resampling...")

    cnt = resample_cnt(cnt, 250.0)

    log.info("Highpassing...")


    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)


    picks = [cnt.info['ch_names'].index(channel) for channel in C_sensors if channel in cnt.info['ch_names']]

    # Seleziona solo i canali desiderati dal Raw object
    cnt_selected = cnt.copy().pick(picks)

    # Log per indicare che la standardizzazione sta iniziando
    log.info("Standardizing...")

    # Esegui la standardizzazione solo sui canali selezionati
    cnt_selected_data = cnt_selected.get_data()

    cnt_selected_data = exponential_running_standardize(cnt_selected_data.T, 
                                                    factor_new=1e-3, 
                                                    init_block_size=1000, 
                                                    eps=1e-4).T

    # Aggiorna i dati nel Raw object `cnt` solo per i canali selezionati
    cnt._data[picks, :] = cnt_selected_data

    ival = [-500, 4000]
    

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)

    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    dataset.X = dataset.X[:, :-1]
    
    
    # Normalizzazione channel-wise
    if subject != 14:
        print("[INFO] Normalizing channel-wise for subject {}...".format(subject))
        dataset.X = (dataset.X - dataset.X.mean(axis=2, keepdims=True)) / (dataset.X.std(axis=2, keepdims=True) + 1e-5)
    else:
        print("[INFO] Skipping normalization for subject 14.")
        
    # Rumore gaussiano + sinusoidi + scaling
    if training and subject != 14:
        print("[INFO] Adding standard augmentation for subject {}...".format(subject))
        dataset.X += np.random.normal(0, 0.01, dataset.X.shape)
        dataset.X += 0.005 * np.sin(2 * np.pi * np.random.rand(*dataset.X.shape))
        scaling = np.random.uniform(0.9, 1.1, (dataset.X.shape[0], 1, 1))
        dataset.X *= scaling
    elif subject == 14:
        print("[INFO] Skipping augmentation for subject 14.")
    
    return dataset.X, dataset.y
