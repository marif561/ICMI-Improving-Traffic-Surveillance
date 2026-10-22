# Data

## Road Anomaly Dataset (RAD)

The dataset introduced with the paper is published on Zenodo:

> Sarfaraz Natha. *Comprehensive Dataset for Detecting Road Anomalies in Diverse Real-World Situations.* Zenodo, 2024. [doi:10.5281/zenodo.13832363](https://doi.org/10.5281/zenodo.13832363) · CC BY 4.0

The dataset contains road surveillance images and videos collected with mobile and CCTV cameras in Pakistan. Classes:

| Class | In the paper | In the Zenodo release |
|---|:-:|:-:|
| Vehicle accident | ✓ | ✓ |
| Fighting | ✓ | ✓ |
| Vehicle / car fire | ✓ | ✓ |
| Snatching at gunpoint | ✓ | ✓ |
| Potholes | | ✓ |

The Zenodo archive (`Road_Anomaly_Dataset.zip`, about 1.8 GB) also has YOLO-format bounding boxes. This code only uses the class of each image.

## Expected layout

Classification expects one folder per class:

```
data/rad/
├── accident/   *.jpg
├── car_fire/   *.jpg
├── fighting/   *.jpg
└── snatching/  *.jpg
```

If the root contains `train/`, `val/` and `test/` folders with the same class subfolders, those splits are used. Otherwise a stratified 70/15/15 split is made with seed 42.

To start from videos, place them as `data/videos/<class>/*.mp4` and run:

```bash
python -m roadanomaly.frames --videos data/videos --out data/rad --every 5
```

Data folders are git-ignored. Don't commit the dataset; link to Zenodo instead.
