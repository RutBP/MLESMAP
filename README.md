# A Comparative Study of Machine-Learning Strategies for Ground Shaking Maps Applied to Southern Iceland


## Overview

Ground-motion prediction is formulated as a **supervised regression problem**, where earthquake source parameters, source-to-site geometry, and spatial coordinates are used to estimate the corresponding ground-motion intensity for a given earthquake.

In this case the target variable is:

* **RotD50** (cm/s²)

To improve model stability and account for the highly skewed distribution of seismic ground-motion values, the model is trained using the **natural logarithm of RotD50**:
Predictions are transformed back to physical units (cm/s²) during post-processing and map generation when required.

---

## Repository Structure

```text
.
├── 1_GridSearch.py 
├── 2_Train.py
├── 3_Validate.py
└── 4_Inference.py
└── _search.py
└── _validation.py
└── minmax_scaler.py
```


## Dataset Description

The Random Forest model uses earthquake source information and source-to-site geometry as predictor variables.

### Input Features

| Feature            | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `Source_ID`        | Unique identifier of the seismic event                   |
| `Site_Lat`         | Latitude of the synthetic recording site                 |
| `Site_Lon`         | Longitude of the synthetic recording site                |
| `Hypocenter_Lat`   | Latitude of the synthetic earthquake hypocenter          |
| `Hypocenter_Lon`   | Longitude of the synthetic earthquake hypocenter         |
| `Hypocenter_Depth` | Hypocentral depth (km)                                   |
| `Magnitude`        | Earthquake magnitude                                     |
| `Distance`         | Source-to-site distance (km)                             |
| `Azimuth`          | Azimuth from source to site                              |




---
## Workflow

1. Run `1_GridSearch.py` multiple times with different hyperparameter combinations to find the optimal configuration.
2. Train the model with `2_Train.py`.
3. Validate the model with `3_Validate.py`.
4. Run `4_Inference.py` by specifying the earthquake latitude, longitude, depth and magnitude.

---

## Requirements

The workflow is implemented using **dislib**, a distributed machine-learning library designed for large-scale data analytics on High-Performance Computing (HPC) systems. Dislib provides scalable implementations of machine-learning algorithms and operates on top of the PyCOMPSs programming model, enabling distributed execution across multiple computing resources.

Installation instructions for dislib and PyCOMPSs can be found in the official documentation (https://compss.readthedocs.io/en/latest/Sections/01_Installation_Configuration.html).

To run the workflow successfully, the files `_validation.py`, `_search.py`, and `minmax_scaler.py` must be located in the same directory as the main workflow scripts (`1_GridSearch.py`, `2_Train.py`, `3_Validate.py`, and `4_Inference.py`), since they are imported as local dependencies during execution.


---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Data Availability

The datasets used in this project are available upon request. Please contact the authors for further information regarding data access.

---

## Contact

For questions regarding the methodology, implementation, or data used in this study, please contact rut.blanco@bsc.es.
