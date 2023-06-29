# DFDC dataset customization for DSFD and RetinaNet
This is a customization of [the implementation of DSFD and RetinaNet](https://github.com/hukkelas/DSFD-Pytorch-Inference.git). This works on datasets that have same structure like the [DFDC dataset](https://www.kaggle.com/c/deepfake-detection-challenge).  
_THIS CODE **DOES NOT** HANDLE ERRORS._

0. Install independencies:
```pip install -r requirements.txt```

1. Find bounding boxes:  
This takes in videos and return bounding boxes in .json file.  
```python detect_faces.py [--argument value [...]]```  
For help: ```python detect_faces.py --help```

2. Crop frames according to bounding boxes:  
This reads the bounding boxes from .json file and crop the video frames.  
```python crop_frames.py [--argument value [...]]```  
For help: ```python crop_frames.py --help```