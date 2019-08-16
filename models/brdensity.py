#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:46:10 2019

Apply ML/DL models to files queried in OrthancBDRAD3

@author: Yesh
"""
import numpy as np
import cv2
from keras.models import load_model

from utils import update_ML_dicom, upload_dicom_file, get_series
from utils import add_labels,  update_pixels, read_dcm

def is_mammo_study(study):
    if study['MainDicomTags'].get('StudyDescription'):
        if 'MAM' in study['MainDicomTags'].get('StudyDescription'):
            return True
    else:
        return False

def is_series_2d_mammo(series):
    if (series['MainDicomTags'].get('Modality').lower() == 'mg' and
        'breast' in series['MainDicomTags'].get('BodyPartExamined').lower()):
        try:
            if 'tomo' in series['MainDicomTags'].get('SeriesDescription').lower():
                return False
        except AttributeError:
            pass

        return True
    else:
        return False


def create_output_img(ds, text_lines):

    img_bg = np.zeros([ds.Rows, ds.Columns], dtype=np.uint8)
    img_bg.fill(0) # rbg color

    xy = (ds.Rows//30,  ds.Columns//5)

    img = add_labels(img_bg, text_lines, fontsize=ds.Columns//30, xy=xy)
    img = img // img.max() * int(ds.pixel_array.max()) # set to the text pixel value to the max of the mammo

#    import matplotlib.pyplot as plt
#    plt.imshow(img)
#    plt.savefig(fname='temp/test.png',dpi=600)

    return img


###################################################
### Classification of breast density on mammography
###################################################

def model(study, ml_series_uid):
    model_name = 'BI-RADS Density Demo'
    print('Apply BI-RADS Breast Density Classifiction to study')

    # 1. Check headers
    if is_mammo_study(study) == False:
        return []

    # 2. Load Model
    def dummy_model(img):
        return [0, 0, 0, 0, 1]
    h, w, c = 224, 224, 1

    y_preds = []
    instance_names = []
    instance_labels = []
    instance_conf = []
    density_labels = ['A', 'B', 'C', 'D', 'TEST DENSITY']

    responses = []
    for series_id in study['Series']:
        series = get_series(series_id)

        # 3. check if series is relevant to model
        if is_series_2d_mammo(series) == False:
            continue
        print('Analyzing series: {}'.format(series_id))


        for i, instance_id in enumerate(series['Instances']):
            ds, img = read_dcm(instance_id)

            # Custom Algorithm and Modifications
            img = cv2.resize(img, (h, w))
            stack = [img for i in range(c)]
            img = np.stack(stack, axis=-1)
            img = (img - np.mean(img)) / np.std(img)
            img = np.expand_dims(img, axis=0)

            y_pred = dummy_model(img)

            pred_idx = np.argmax(y_pred)
            
            instance_names.append(ds.SeriesDescription)
            instance_labels.append(density_labels[pred_idx])
            instance_conf.append(y_pred[pred_idx])

            y_preds.append(y_pred)


    y_preds = np.asarray(y_preds)
    y_preds_mean = np.mean(y_preds, axis=0)


    preds_idx = np.argmax(y_preds_mean)
    preds_conf = y_preds_mean[preds_idx]
    preds_label = density_labels[preds_idx]

    text_lines = []
    text_lines.append('Model: {}'.format(model_name))
    text_lines.append('Breast Density: {}'.format(preds_label))
    text_lines.append('Confidence: {0:.0%}'.format(preds_conf))
    text_lines.append('NOT FOR CLINICAL USE')
    text_lines.append('FOR RESEARCH ONLY')
    text_lines.append('-'*50)

    for name, label, conf in zip(instance_names, instance_labels, instance_conf):
        text_lines.append('{0} View: Density {1} (confidence: {2:.0%})'.format(name, label, conf))

    # create summary image
    img = create_output_img(ds, text_lines)

    # update pixels and metadata using a random dcm from study as a template
    ds = update_ML_dicom(ds, model_name, ml_series_uid)
    ds = update_pixels(ds, img)

    # upload & append
    r = upload_dicom_file(ds) # upload new edited dicom
    responses.append(r)

    return responses

