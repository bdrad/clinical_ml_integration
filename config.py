#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config File
"""

REMOTE = 'PACS' # specify PACS alias where ML analysis will be sent. This should match alias in Configuration.json
A = False # Set to True to automaticall transmit to remote
DELETE_AFTER_TRANSMIT = False # Set to True to automatically delete study after transmitting analysis to the remote


ORTHANC = 'http://localhost:8042/' # Where to look for Orthanc
ML_SERIES_DESCRIPTION = 'ML_models' # identifying description for DICOM series containing all ML DICOM outputs