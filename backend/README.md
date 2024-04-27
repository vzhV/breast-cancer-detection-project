# Breast Cancer Detector API

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
