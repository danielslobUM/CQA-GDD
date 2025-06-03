# Scripts required for continuous quality assurance of deep learning segmentaions in radiotherapy.

Welcome! This repository contains a collection of Python scripts used in "Results of continuous quality assurance for deep learning segmentations in radiotherapy making use of statistical process control for automatic performance monitoring" by Van Acht et al. 
These scripts cover a variety of tasks.

## 📂 Scripts

| File Name       | Description                          |
|-----------------|--------------------------------------|
| `SPC_control_limits.py` | Script that determines the control limits based on SPC. Required to perfrom automatic monitoring, not for visualisation in GUI | 
| `CQA_DLS_GUI.py`   | GUI that lets you score data, create plots, make datasheets and detect outliers and trend shifts.|
| `GUI_functions.py`    | Functions required for all functionalities in the GUI            |
| `calculator_functions.py`      | Functions required to get scores from RTstruct.dcm files, integrated in GUI   |


### Determining the Control Limits

With `SPC_control_limits.py` the control limits can be determined required for statistical process controll (SPC). This script will ask for some input values to create control_limits.json. This file will later be used for the SPC plots in the GUI. As mentioned in the article. One should consider when to update the control limits and do this periodically, not continuously. 

### Using the GUI

With `CQA_DLS_GUI.py` a GUI will help scoring your data, visualise it and even perform monitoring using Nelson rules

![Appearance of GUI when `CQA_DLS_GUI.py` is run](GUI.PNG)


## 💻 Requirements

- Python 3.8.7
- The following Python packages (exact versions recommended):
  - Numpy 1.23.5
  - Scipy 1.10.1
  - Matplotlib 3.7.5
  - Pydicom 2.4.4
  - surface_distance 0.1

```bash
pip install -r requirements.txt
```
> ⚠️ This project was developed and tested with **Python 3.8.7**. Using other versions might work but is not guaranteed.
>

## 🛠️ Setup

Although this code is written to work for other institute than the development institute, this still requires some tailoring of the data collection and personalisation of the code.

### Data Collection

The clinical segmentations (CS) and deep learning segmentatation (DLS) RTstruct DICOM files need to be exported from a treatment planning stystem to a local drive which you can access from your preferred code editor. The code assumes that the same names are used for the ROIs in CS and DLS. It will automatically only consider availabel pairs. If there is no pair between CS and DLS, it will not be considered by the script. Also the file names of these RTStruct.dcm files needs to be very specific. In the institute where this code is developed it was chosen to do type%pid%CTid%date%time%protocol.dcm. This would result in a pair of CS - DLS DICOM files that can be linked to one another. In the code (calculator_functions.py) this can be found in the get_scores function. Here it can be seen that AS (auto-segmentation and thus DLS) and MS (manual segmentation and thus CS) files are linked within the patient folder. As example a CS RTStruct.dcm file is named: AS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm. The corresponding DLS RTstruct.dcm file is named: MS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm. You can change the AS and MS to DLS and CS respectively for consistency. If you do this, also change it in calculator_functions.py.

These DICOM files are saved on data folder. This data folder has a very specific setup. In the parent data folder there will be folders for every day an DLS RTStruct is saved, a day folder. In this day folder there will be folders for every patient that had a DLS RTStruct saved that day, a patient folder. In this patient folder the RTStruct DICOM is saved. Later when the CS RTStruct DICOM is saved, it should be saved in this same patient folder. Eventually the scores.npy file will also be saved in this patient folder. Considering the same examples as above, the path to the folder will be "path to parent data folder"\20240101\1234567890\. In here will be three files:
-  AS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm (or DLS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm if you want to change up the name)
-  MS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm (or CS%1234567890%1.2.34.5.6.7.8.9.1233456789%20240101%090000.dcm if you want to change up the name)
-  scores_1234567890_1.2.34.5.6.7.8.9.1233456789.npy (CT id is mentioned as it can occurr that multiple RTStrutcs are exported based on different CTs)

### Code Personalisation

To make the scripts work a few adjustments are required in the `CQA_DLS_GUI.py` script:
- self.data_path: should be changed to your parent data folder
- self.save_path: should be changed to the folder where you want to save all results
- self.control_limits_file: should be changed to the full string of the location of the control_limits.json file (created by `SPC_control_limits.py` script)
- self.exclusion_path: an .csv file should be created with the ROIs you want to exclude from CQA (like CouchTop, LaserLoc or any other exported ROIs). This path refers to this .csv file.
- self.N_min: SPC should only be considered after a certain amount of data points are present. This N_min represents that value (10 is considered a starting point by Van Acht et al.). Use the same value when determining the control limits.
- self.xdim, self.ydim, self.zdim: the dimension of a voxel of the CT scan. Since we are not exporting CT scans we can not automatically get these values. Put them in manually.
- self.value1, self.value2, self.value3: values used by the adapted Nelson rules to detect outliers, trend shifts and trend drifts. Further explained in Van Acht et al.

