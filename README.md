The RSNA Abdominal Trauma Detection AI Challenge involves developing AI models to detect and grade severe injuries in internal abdominal organs (liver, spleen, kidneys, bowel) and identify active internal bleeding using computed tomography (CT) scans.

[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

# Installation

 1. Conda environment

 ```bash
 conda env create --file environment.yml
 ```

 2. Activate environment

  ```bash
 conda activate rsna_atd
 ```

 3. Install package through pip

  ```bash
pip install .
 ```

 NOTE: If you want to modify some of the codebase, you should pip install with the `-e` flag.

# Objective

Each patient can have one or multiple abdominal CT scans. The prediction is made per patient, indicating the corresponding trauma (or lack of).

![sample_image.png](assets\sample_image.png)

# Data

1. **train.csv:** This file provides target labels for the training set. Each patient has a unique ID code, and injury types (bowel/extravasation) are paired with health states (healthy/injury), as well as organ injury levels (kidney/liver/spleen) with different severity levels (healthy/low/high). The "any_injury" column indicates whether the patient had any injury.

| Column | Description |
| --- | --- |
| patient_id (PK) | Unique ID code for each patient. |
| bowel_[healthy/injury] | Binary target for bowel injury (healthy/injury). |
| extravasation_[healthy/injury] | Binary target for extravasation injury (healthy/injury). |
| kidney_[healthy/low/high] | Target levels for kidney injury (healthy/low/high). |
| liver_[healthy/low/high] | Target levels for liver injury (healthy/low/high). |
| spleen_[healthy/low/high] | Target levels for spleen injury (healthy/low/high). |
| any_injury | Indicator whether the patient had any injury. |

2. **[train/test]_series_meta.csv:** This file contains metadata about each scan. It includes patient IDs, series IDs for scans, aortic_hu (indicating scan timing), and whether the scan covered all organs (incomplete_organ label).

| Column | Description |
| --- | --- |
| patient_id (FK) | Unique ID code for each patient. |
| series_id (PK) | Unique ID code for each scan. |
| instance_number | Image number within the scan. |
| injury_name | Type of injury visible in the frame. |

3. **[train/test]_images/[patient_id]/[series_id]/[image_instance_number].dcm:** These are the CT scan data files in DICOM format. Different CT machines were used, leading to variations in pixel properties and formats. Each patient may have undergone one or two scans.

4.  **image_level_labels.csv:** This file, present only in the training set, identifies images containing bowel or extravasation injuries. It includes patient IDs, series IDs, instance numbers (image order), and injury names.

| Column | Description |
| --- | --- |
| patient_id | Unique ID code for each patient. |
| series_id | Unique ID code for each scan. |
| instance_number | Image number within the scan. |
| injury_name | Type of injury visible in the frame. |

5. **sample_submission.csv:** This file provides a sample submission format for predictions.
6. **segmentations/:** This directory contains model-generated pixel-level annotations of relevant organs and bones for a subset of training scans in NIFTI format. Each filename corresponds to a series ID.
7. **[train/test]_dicom_tags.parquet:** DICOM tags extracted using Pydicom, provided for convenience.

```markdown
[train/test]_images/
+-- [patient_id]/
    +-- [series_id]/
        +-- [image_instance_number_1].dcm
        +-- [image_instance_number_2].dcm
        +-- ...
        +-- [image_instance_number_n].dcm

train.csv               train_series_meta.csv
+--------------+        +-------------------------+
| patient_id   | -----> | patient_id              |
| bowel_healthy|        | series_id               |
|     ...      |        | aortic_hu               |
| any_injury   |        | incomplete_organ        |
+--------------+        +-------------------------+
```

# Code structure

```markdown
rsna_atd
├── config.py               # Configuration parameters
├── data.py                 # Data processing tools
├── models.py               # Model definitions
├── utils.py                # Miscelaneous utilities
└── visulization.py         # Visualization

```