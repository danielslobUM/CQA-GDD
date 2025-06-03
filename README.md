# Scripts required for continuous quality assurance of deep learning segmentaions in radiotherapy.

Welcome! This repository contains a collection of Python scripts used in "Results of continuous quality assurance for deep learning segmentations in radiotherapy making use of statistical process control for automatic performance monitoring" by Van Acht et al. 
These scripts cover a variety of tasks.

## 📂 Scripts

| File Name       | Description                          |
|-----------------|--------------------------------------|
| `CQA_DLS_GUI.py`   | GUI that lets you score data, create plots, make datasheets and detect outliers and trend shifts.|
| `GUI_functions.py`    | Functions required for all functionalities in the GUI            |
| `caculator_functions.py`      | Functions required to get scores from RTstruct.dcm files, integrated in GUI   |
| `SPC_control_limits.py` | Script that determines the control limits based on SPC. Required to perfrom automatic monitoring, not for visualisation in GUI | 


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

Although this code is written to work for other institute than the development institute, this still requires some tailoring of the data setup and tailoring of the code.

### Data

The data needs to be exported from a treatment planning stystem to a local drive which you can access from your preferred code editor.
