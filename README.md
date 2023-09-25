# EfficientVIT for Segmentation 
  
  
# Requirements
  python 3.7/3.8 + pytorch 1.9.0 (biult on [EGFNet](https://github.com/ShaohuaDong2021/EGFNet))
   
   
# Segmentation maps and performance
   We provide segmentation maps on MFNet dataset and PST900 dataset under './model/'.
   
   **Performace on MFNet dataset**
   
   
   **Performace on PST900 dataset**
   
 

# Training
1. Download [MFNet dataset](https://pan.baidu.com/s/1NHGazP7pwgEM47SP_ljJPg) (code: 3b9o) or [PST900 dataset](https://pan.baidu.com/s/13xgwFfUbu8zNvkwJq2Ggug) (code: mp2h).
2. Run train_base_vit.py (default to MFNet Dataset).

Note: our main model is under './toolbox/models/LASNet_base.py'


# Pre-trained model and testing

1. Rename the name of the pre-trained model to 'model.pth', and then run test_base_vit.py (default to MFNet Dataset).
  
  
# Citation
       
                
