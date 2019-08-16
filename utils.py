#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:10:06 2019

@author: Yesh
"""
# import os; os.chdir('/Users/Yesh/Documents/BDRAD/clinical_integration')
import os
import numpy as np
import requests
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
from PIL import Image, ImageDraw, ImageFont


BASE = 'http://localhost:8042/'
print(BASE)

## BASIC GET FUNCTIONS
def get_all_patient_ids():
    return requests.get(BASE + 'patients').json()

def get_patient(pt_id):
    return requests.get(BASE + 'patients/' + pt_id).json()

def get_all_study_ids():
    return requests.get(BASE + 'studies').json()

def get_study(study_id):
    return requests.get(BASE + 'studies/' + study_id).json()

def delete_study(study_id):
    return requests.delete(BASE + 'studies/' + study_id).json()

def get_all_series():
    return requests.get(BASE + 'series').json()

def get_series(series_id):
    return requests.get(BASE + 'series/' + series_id).json()

def get_series_in_study(study_id):
    series_all = []
    series_ids = get_study(study_id)['Series']
    for series_id in series_ids:
        series_all.append(get_series(series_id))
    return series_all

def get_all_instances():
    return requests.get(BASE + 'instances').json()

def get_instance(instance_id):
    return requests.get(BASE + 'instances/' + instance_id).json()

def delete_instance(instance_id):
    return requests.delete(BASE + 'instances/' + instance_id).json()

def get_changes(arg = ''):
    # arg = 'last', 'limit=N', 'since'
    return requests.get(BASE + 'changes?' + arg).json()

def post_instance(dcm):
    return requests.post(BASE + 'instances', data=dcm)

def get_instance_dcm(instance_id):
    # get raw pixel and wrap into a dcmread-friendly format with DicomBytesIO
    r = requests.get(BASE + 'instances/' + instance_id + '/file').content
    return dcmread(DicomBytesIO(r))

def get_series_dcms(series_id):
    instance_ids = get_series(series_id)['Instances']
    ds_list = []
    for instance_id in instance_ids:
        ds = get_instance_dcm(instance_id)
        ds_list.append([ds.InstanceNumber, ds, instance_id])
    return sorted(ds_list, key=lambda tup: tup[0])

def get_new_uid(level):
    return requests.get(BASE + 'tools/generate-uid?level='+level).content.decode('utf-8')


def upload_dicom_file(ds):
    fp_dcm = './temp/' + ds.SOPInstanceUID + '.dcm'
    ds.save_as(fp_dcm)

    f = open(fp_dcm, "rb")
    content = f.read()
    f.close()
    r = post_instance(content).json()
    os.remove(fp_dcm)
    return r

def transmit_file(uuid, remote='MAM'):
    return requests.post(BASE + 'modalities/{}/store'.format(remote), data=uuid)

def read_dcm(instance_id):
    # save dicom and read from saved location.
    # Otherwise pydicom gives an image not read error
    ds = get_instance_dcm(instance_id)
    fp_dcm = './temp/' + instance_id + '.dcm'

    ds.save_as(fp_dcm)
    ds = dcmread(fp_dcm)

    # remove path after loading pixel_array. Otherwise casue NotImplementedError
    img = ds.pixel_array
    os.remove(fp_dcm)

    return ds, img





####################################
# UID GENERATORS AND FETCHERS
# AND UPDATING DICOM
####################################
ml_series_description = 'ML_models'

def get_ML_series(study_id, ml_series_description=ml_series_description):
    # check if any AI series already exists with match Series Description
    # if not, gen new uid and return that
    series_all = get_series_in_study(study_id)
    ds = get_instance_dcm(series_all[0]['Instances'][0])


    for series in series_all:
        try:
            if ml_series_description in series['MainDicomTags']['SeriesDescription']:
                ds = get_instance_dcm(series['Instances'][0])
                return ds.SeriesInstanceUID
        except:
            pass

    return get_new_uid('series')



def update_ML_dicom(ds, model_name, ml_series_uid, ml_series_description=ml_series_description):
    # set model name
    ds.ManufacturerModelName = model_name

    ds.SeriesDescription = ml_series_description
    ds.SeriesInstanceUID = ml_series_uid

    ds.SOPInstanceUID = get_new_uid('instance')

    # zero out some others DICOM headers
    ds.ViewCodeSequence = ''
    ds.PatientOrientation = ''
    ds.ProtocolName = ''
    ds.ViewPosition = ''

    return ds


def update_pixels(ds, pixel_array_mod):
    if ds.pixel_array.flags['WRITEABLE'] == False:
        ds.pixel_array.setflags(write=1)


    # Decompress before editing pixel_array & PixelData.
    # otherwise will cause an issue when saving
    ds.decompress()

    # edit pixel_array AND PixelData!
    ds.pixel_array[:,:] = pixel_array_mod[:,:]
    ds.PixelData = ds.pixel_array.tobytes()

    return ds


#############################
# EDIT DICOM
#############################
def add_labels(pixel_array, lines, fontsize=None, xy=None, fill='white'):
    # adds a text label to dicom image
    if fontsize == None:
        fontsize = pixel_array.shape[0]//30
    if xy == None:
        xy = (pixel_array.shape[1]//4,  pixel_array.shape[0]//30)

    img = Image.fromarray(pixel_array)
    d = ImageDraw.Draw(img)
    font_file = 'arial.ttf' # may need to change if on a different computer
    font = ImageFont.truetype(font_file, fontsize)

    for text in lines:
        d.text(xy, text, font=font, fill=fill)
        xy = (xy[0], xy[1] + fontsize)

    img_array = np.array(img)

    return img_array
