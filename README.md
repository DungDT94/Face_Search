# Face_Search
This repository utilize insightface to detect and extract face feature, hnswlib to search face


## Installation
Install the required python library, recommend using anaconda enviroment python >= 3.8 to avoid conflict. In terminal
```
conda create --name <your_env> --file <path to requirements.txt>
```

## Usage
Download weight in below link:
```
https://drive.google.com/file/d/1a2fDjRMgE2s7ctH468p3I7Tt4MywGdac/view?usp=drive_link
```
```
https://drive.google.com/file/d/1szWEeDq7ddRPBH-a6K5HW-ZtGfz75yoW/view?usp=sharing
```
Replace path of weight in file ```config.py```

### Inference
Run file ```app.py``` to test model, this model take cv2 img.

### Deploy
Run file ```deploy.py``` to deploy to your local machine.
#### Test by Post Man:
##### Search Image
##### Add Image

