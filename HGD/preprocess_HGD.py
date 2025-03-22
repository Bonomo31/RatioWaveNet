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
#from braindecode.datautil.signalproc import exponential_running_standardize
#from braindecode.datautil.signalproc import highpass_cnt

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

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [2]), ('Left Hand', [4],),
                              ('Rest', [6]), ('Feet', [8])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    #print("Set For Cleaning:",type(set_for_cleaning))
    #print("Attributi: ",dir(set_for_cleaning))
    #print(set_for_cleaning.X.shape)
    #if hasattr(set_for_cleaning, 'X'):
    #  print(set_for_cleaning.X.shape)
    #else:
    #  print("L'attributo 'X' non esiste in 'set_for_cleaning'.")

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))
    
    #print("Numero di trials validi:", np.sum(clean_trial_mask))


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
    
    #print("Dopo aggiunta canale di stimolazione: ",cnt.ch_names)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    """log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    print("Dopo highpass: ", cnt.get_data().shape)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)
    print("Dopo standardizzazione: ", cnt.get_data().shape)"""

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]
    
    #Stampiamo i dati che passiamo alla funzione create_signal_target_from_raw_mne
    #print("Dati passati a create_signal_target_from_raw_mne:")
    #print("cnt: ",type(cnt_1))
    #print("marker_def: ",type(marker_def))
    #print("ival: ",type(ival))
    #print("cnt: ",cnt_1)
    #print("marker_def: ",marker_def)
    #print("ival: ",ival)

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    #Ultimo canale è il canale di stimolazione e non è necessario quindi lo rimuoviamo
    dataset.X = dataset.X[:, :-1]
    print("Dopo pulizia: ", dataset.X.shape)
    print("Dopo pulizia: ", dataset.y.shape)
    return dataset.X, dataset.y
