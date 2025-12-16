import logging
from copy import deepcopy

import resampy
from mne.io.base import concatenate_raws
import mne
import numpy as np

log = logging.getLogger(__name__)


def concatenate_raws_with_events(raws):
    """
    Concatenates `mne.io.RawArray` objects, respects `info['events']` attributes
    and concatenates them correctly. Also does not modify `raws[0]` inplace
    as the :func:`concatenate_raws` function of MNE does.
    
    Parameters
    ----------
    raws: list of `mne.io.RawArray`

    Returns
    -------
    concatenated_raw: `mne.io.RawArray`
    """
    # prevent in-place modification of raws[0]
    raws[0] = deepcopy(raws[0])
    event_lists = [r.info['events'] for r in raws]
    new_raw, new_events = concatenate_raws(raws, events_list=event_lists)
    new_raw.info['events'] = new_events
    return new_raw


import mne
import numpy as np
import resampy
from copy import deepcopy

def resample_cnt(cnt, new_fs):
    """
    Resample continuous EEG recording using resampy.

    Parameters
    ----------
    cnt : `mne.io.Raw`
        EEG recording.
    new_fs : float
        New sampling rate.

    Returns
    -------
    resampled : `mne.io.Raw`
        Resampled EEG object.
    """
    old_fs = cnt.info['sfreq']

    if new_fs == old_fs:
        print("Nessun resampling necessario, le frequenze sono uguali.")
        return deepcopy(cnt)

    print(f"Resampling da {old_fs:.2f} Hz a {new_fs:.2f} Hz...")

    # Stampa la frequenza di campionamento prima del resampling
    print("sfreq prima del resampling:", cnt.info['sfreq'])

    # Resampling in-place
    cnt.resample(new_fs)

    # Stampa la frequenza di campionamento dopo il resampling
    print("sfreq dopo il resampling:", cnt.info['sfreq'])

    # Controllo se il resampling è stato effettuato correttamente
    if cnt.info['sfreq'] != new_fs:
        raise ValueError(f"Resampling fallito: {cnt.info['sfreq']} Hz")
    else:
        print(f"Resampling effettuato con successo: {cnt.info['sfreq']} Hz")

    # Controllo la forma dei dati
    print("Forma dei dati dopo il resampling:", cnt.get_data().shape)

    return cnt 



def mne_apply(func, raw, verbose='WARNING'):
    """
    Apply function to data of `mne.io.RawArray`.
    
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)
