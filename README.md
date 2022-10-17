# Clinical Integration of Machine Learning Models in Hospital PACS
This repository contains the template code for the proposed clinical machine learning integration architecture in paper “An open source, vender agnostic hardware and software pipeline for integration of artificial intelligence in radiology workflow.” This code is a demonstration of how to integrate machine learning models in any existing hospital PACS, utilizing an DICOM/GPU server running Orthanc. The Python code in this repo watches Orthanc for new studies and applies relevant machine learning models. We’ve included a template for how to integrate the BI-RADS breast density classification model described in the paper, but, due to data privacy restrictions, you will need to supply your own model weights.

Please see the paper for further details.

## Getting Started

### General Architecture


This ReadMe will explain how to setup the Orthanc DICOM Server and steps 4-8 from the above figure.

### Orthanc Setup & Configuration
Please follow instructions at https://www.orthanc-server.com to install the Orthanc server.

Generate a documented configuration file
```bash
cd Orthanc
Orthanc --config=Configuration.json
```

Modify DICOM modalities in Configuration.json section of the configuration file to include your institutions PACS system. You will need the PACS alias, IP address, ports, AET titles, and the relevant vendor patch parameter (optional).

The Orthanc server was configured by specifying the PACS alias, IP address, ports and Application Entity Titles, and a vendor patch parameter in the configuration file
### Download and Configure Clinical Integration Repository
1. Clone this repository
```
git clone https://github.com/bdrad/clinical_ml_integration.git
cd clinical_ml_integration
```

2. Create environment and install dependencies
```bash
conda env create -f requirements.yml
source activate clinical_integration
```

3. Edit config.py
```python
remote = 'PACS' # specify PACS alias where ML analysis will be sent. This should match alias in Configuration.json
auto_transmit = False # Set to True to automaticall transmit to remote
delete_after_transmit = False # Set to True to automatically delete study after transmitting analysis to the remote
```

## Starting the Clinical ML Integration Server
In one terminal start Orthanc with configuration file
```bash
Orthanc ./Configuration.json
```

In another terminal run clinical_ml_integration repo
```bash
cd clinical_ml_integration
python main.py
```

## Uploading Custom Models
This repo contains template code for a BI-RADS breast density classification algorithm that was used to demo the clinical ML integration framework in the paper. Because of data privacy restrictions the model weights cannot be published. However, we explain below how to implement your own model.

### Basics & Required Files
To add a custom ML model, you need:
1. Download any extra packages required to run the model
2. Save your weights to `models_weights/` directory
3. Upload the model analysis python script (described below) to the `models/` directory

### Model Analysis Script
`models/brdensity.py` is an example of a model analysis script. `main.py` will load all scripts in the `models/` directory. It must contain the following steps:

1. Import helper functions
```python
from utils import update_ML_dicom, upload_dicom_file, get_series
from utils import add_labels,  update_pixels, read_dcm
```

1. Define model function that accepts an Orthanc study and ml_series_uid as parameters. **Function must be named "model".** Initialize `responses` list, which model function will return.
```python
def model(study, ml_series_uid):
    # study = Orthanc Study object (https://book.orthanc-server.com/users/rest.html#browsing-from-the-patient-down-to-the-instance)
    # ml_series_uid = Series UID that holds instances of machine learning model outputs.
    responses = []
    ...
```

1. Check DICOM headers to determine if you should apply this algorithm to the study. **IMPORTANT: Be very specific. If you skip this step or are not specific, the algorithm will get applied to every study/many studies**
```python
  def is_relevant_study(study):
      if study['MainDicomTags'] == ...
          return True
      else:
          return False

  def model(study, ml_series_uid):
      responses = []
      if is_relevant_study(study) == False: # check DICOM header information
          return responses
```

1. Load model weights
```python
  from keras.models import load_model

  def model(study, ml_series_uid):
      responses = []
      if relevant_study == False: # check DICOM header information
          return responses # must return empty array if not relevant model

      model = load_model(PATH_TO_WEIGHTS) # load model using relevant package (keras, sklearn, pickle, etc.)
```

1. Loop through each series and check which series are relevant to model (ML model may not be appropriate for every series in study).
```python
  def model(study, ml_series_uid):
      responses = []
      if relevant_study == False:
          return responses
      model = load_model(PATH_TO_WEIGHTS)

      for series_id in study['Series']:
          series = get_series(series_id) # helper function from utils.py
          if is_series_relevant(series) == False:
              continue
```

1. Loop through each instance of the series and apply preprocessing, model inference, and postprocessing. Use helper function `read_dcm` to read instance (loads DICOM and DICOM image in way to prevent errors when saving)
```python
  def model(study, ml_series_uid):
      responses = []
      if relevant_study == False:
          return responses
      model = load_model(PATH_TO_WEIGHTS)

      preds_processed = []
      for series_id in study['Series']:
          series = get_series(series_id)
          if is_series_relevant(series) == False:
              continue

          # Loop though instances and apply model
          for instance_id in series['Instances']:
              ds, img = read_dcm(instance_id) # helper function from utils.py
              img = preprocess(img)

              pred = model.predict(img)

              # postprocessing to convert output into legible string
              ## Example: pred_processed = 'Series 1: Class B (89%)'
              pred_processed = postprocess(img)

      # Combine output of each image into a single image
      black_img = np.zeros([ds.Rows, ds.Columns], dtype=np.uint8)
      black_img.fill(0) # rbg color
      output_img = add_labels(pixel_array = black_img, lines=preds_processed) # helper function from utils.py to add multiline text to image      
```

1. Use helper functions `update_ML_dicom`, `update_pixels`, and `upload_dicom_file` to update DICOM metadata. Append output of `upload_dicom_file` to responses and return responses.
```python
  def model(study, ml_series_uid):
      responses = []
      if relevant_study == False:
          return responses
      model = load_model(PATH_TO_WEIGHTS)

      preds_processed = []
      for series_id in study['Series']:
          ...
          for instance_id in series['Instances']:
              ds, img = read_dcm(instance_id)
              ...

      output_img = ...

      # Use last DICOM loaded as template and modify SeriesInstanceUID, SeriesDescription, and SOPInstanceUID and set some other data to ''
      ds = update_ML_dicom(ds, 'model_name', ml_series_uid)

      # Update PixelData on DICOM with helper function from utils.py
      ds = update_pixels(ds, img)

      # Upload new DICOM with ML output image to Orthanc
      r = upload_dicom_file(ds)
      responses.append(r)

      return responses
  ```
  **`responses` must be an empty list or list of values returned from `upload_dicom_file`**

1. **If `auto_upload = True` in config.py, then that's all you need to upload a new model!** If `auto_upload = False`, the script will analyze the image and upload to Orthanc. However, if you want to transmit to your PACS, you must (1) open up Orthanc browser (http://localhost:8042), (2) find the relevant study and DICOM file, (3) use the Orthanc GUI to "Send to remote modality" and select your PACS.

## Administrative/Deployment Tips
- Setup a test PACS environment that mimics your real PACS for testing
- Disable automatic upload and manually test
- Enable auto-deleting to avoid using up max storage
- Be VERY specific with checking DICOM metadata

## Reference
Please cite this repository in projects using this approach, until the paper is published. Once paper is available, please cite the paper.

The paper can be found using the DOI: https://doi.org/10.1007/s10278-020-00348-8
