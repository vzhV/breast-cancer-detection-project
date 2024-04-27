# Breast Cancer Detector Project

## ML part:

### Prerequisites

- [Mammography Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- [Ultrasound Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

### Installation

Clone the repository:

```bash
git clone https://github.com/vzhV/breast-cancer-detection-project.git
cd breast-cancer-detection-project/ml
```

Install the dependencies:

```bash
poetry install
```

### Data preparation

1) Download datasets from the links provided above.
2) In both ultrasound and mammography directories, there's a 'data_preprocessing directory', in which open config.json and adjust it for your needs.
3) Execute preprocess_main.py file


### Training
1) Go to the 'training' directory in both ultrasound and mammography directories.
2) Adjust configs and execute training python scripts.


### Model evaluation
1) Go to the 'eval' directory in both ultrasound and mammography directories.
2) Execute evaluation python scripts. (ATTENTION: you need to have trained models and configured training configs before running evaluation scripts)


## Backend part:

This backend serves as a processing center for mammogram and ultrasound images, providing functionalities such as classification and segmentation.

### Prerequisites

- Python 3.10
- Poetry for Python dependency management

### Installation

Clone the repository:

```bash
git clone https://github.com/vzhV/breast-cancer-detection-project.git
cd breast-cancer-detection-project/backend
```

Install the dependencies:

```bash
poetry install
```

### Setting Up Model Weights

Before you can start the server, you need to place the model weight files into the backend/weights directory. Ensure you have the following models:

mg_class.h5 and mg_segm.h5 for mammogram processing.
us_segm.h5, us_class.pth, and us_class_overlaid.pth for ultrasound processing.

## Running the Server
Start the server using the following command:

```bash
    python main.py
```
The FastAPI server will be available at http://localhost:8000.

## API Endpoints

### Mammography service
- POST /mammogram:  Upload and process mammogram images.

| Parameter | Type  | Description                                 |
|-----------|-------|---------------------------------------------|
| `file`    | `file`| The mammogram image file (PNG, JPEG)        |
| `action`  | `form`| Action to perform: `CLASSIFICATION`, `SEGMENTATION`, `ALL` |

### Ultrasound service
- POST /ultrasound:  Upload and process mammogram images.

| Parameter | Type  | Description                                                                    |
|-----------|-------|--------------------------------------------------------------------------------|
| `file`    | `file`| The ultrasound image file (PNG, JPEG)                                          |
| `action`  | `form`| Action to perform: `CLASSIFICATION`, `SEGMENTATION`, `CLASSIFICATION_OVERLAID` |


## Frontend part:

### Installation

Clone the repository:

```bash
git clone https://github.com/vzhV/breast-cancer-detection-project.git
cd breast-cancer-detection-project/frontend
```

Install the dependencies:

```bash
npm install
```

### Running the app

```bash
npm start
```

The app will be available at http://localhost:3000.

# Citations:

- [CBIS-DDSM Dataset:](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)  Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). **Curated Breast Imaging Subset of DDSM [Dataset]**. The Cancer Imaging Archive. **DOI:**  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY


- [Breast Ultrasound Dataset:](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
