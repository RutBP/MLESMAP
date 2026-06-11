# A Comparative Study of Machine-Learning Strategies for Ground Shaking Maps Applied to Iceland

This repository contains the Python scripts of the Random Forest (RF) workflow developed for the study:

Rapid prediction of ground motion is essential for seismic risk mitigation and emergency response. Traditional ground motion models (GMMs) provide computationally efficient estimates; however, their simplified formulations are often unable to fully capture the complexity of effects related to the source, propagation and site. In this work, we present MLESmap, a machine-learning estimator for ground shaking maps trained on physics-based CyberShake simulations for Southern Iceland. These strategies focus on improving the representativeness and consistency of training data, whilst assessing the robustness of the model under varying conditions regarding the training variables used and the availability of synthetic data, and in scenarios with different spatial coverage. The results demonstrate the potential of physics-based simulations as a complementary alternative to traditional empirical GMMs. When integrated with machine learning models and when synthetic datasets are properly designed, these simulations can match, and even surpass, results comparable to those obtained from empirical GMM, whilst also maintaining high computational efficiency.

---

## Overview

Ground-motion prediction is formulated as a **supervised regression problem**, where earthquake source parameters, source-to-site geometry, and spatial coordinates are used to estimate the corresponding ground-motion intensity for a given earthquake.

The target variable is:

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

The complete workflow should be executed in the following order:

```bash
python 1_GridSearch.py
python 2_Train.py
python 3_Validate.py
python 4_Inference.py
```

---

## Requirements

The workflow is implemented using **dislib**, a distributed machine-learning library designed for large-scale data analytics on High-Performance Computing (HPC) systems. Dislib provides scalable implementations of machine-learning algorithms and operates on top of the PyCOMPSs programming model, enabling distributed execution across multiple computing resources.

Installation instructions for dislib and PyCOMPSs can be found in the official documentation (https://compss.readthedocs.io/en/latest/Sections/01_Installation_Configuration.html).



## Associated Publication

If you use this repository, please cite:

```bibtex
@article{AUTHOR_YEAR,
  title={A Comparative Study of Machine-Learning Strategies for Ground Shaking Maps Applied to Iceland},
  author={Author(s)},
  journal={Journal Name},
  year={YEAR},
  doi={DOI}
}
```

---

## License



---

## Contact

For questions regarding the methodology, implementation, or data used in this study, please contact rut.blanco@bsc.es.
