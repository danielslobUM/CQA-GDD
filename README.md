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


## 🛠️ Requirements

- Python 3.8.7
- The following Python packages (exact versions recommended):

```bash
pip install -r requirements.txt
```
> ⚠️ This project was developed and tested with **Python 3.8.7**. Using other versions might work but is not guaranteed.
