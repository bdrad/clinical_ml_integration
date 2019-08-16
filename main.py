#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file for "Clinical Integration of Machine Learning Models in Hospital PACS"

https://github.com/bdrad/clinical_ml_integration

## COMMANDS
python main.py

"""

from utils import get_changes, get_study, transmit_file
from utils import get_ML_series, delete_study
import config

import time
from keras import backend as K
from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname('./models/'), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


models = {}
for mod in __all__:
    models[mod] = importlib.import_module('models.' + mod).model


print('Imported Models: {}'.format(__all__))




#################################
# MONITOR FOR NEW CHANGES TO PACS
# monitors every 1 second
#################################

# get last change Seq # to set start point of loop
n_last = get_changes('last')['Last']

while True:
    changes = get_changes('since={}'.format(n_last))

    for item in changes['Changes']:
        if item['ChangeType'] == 'NewStudy':
            study = get_study(item['ID'])
            print('-'*50)
            print('PROCESSING STUDY: ' + item['ID'])
            print('Accession: ' + study.get('MainDicomTags').get('AccessionNumber'))
            print('-'*50)

            # wait until study "is stable" (i.e. has not recieved new instances in a while)
            ##Waits 1 min (can be configured) to see if a new instance will upload
            while study['IsStable'] == False:
                study = get_study(item['ID'])

            ml_series_uid = get_ML_series(study['ID'])
            for model in models:
                responses = models[model](study, ml_series_uid)
                K.clear_session()
                for response in responses:
                    if response.get('Status') == 'Success':
                        if config.AUTO_TRANSMIT:
                            transmission = transmit_file(response['ID'], remote=config.REMOTE)

            print('COMPLETED PROCESSING STUDY: ' + item['ID'])
            
            if config.AUTO_TRANSMIT and config.DELETE_AFTER_TRANSMIT:
                delete_study(item['ID'])


    n_last = changes['Last']
    if changes['Done']:
        time.sleep(1)
