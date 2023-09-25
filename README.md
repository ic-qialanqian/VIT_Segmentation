# EfficientVIT for Segmentation 
  
  
# Requirements
  python 3.7/3.8 + pytorch 1.9.0 (biult on [EGFNet](https://github.com/ShaohuaDong2021/EGFNet))
   
   
# Segmentation maps and performance
   We provide segmentation maps on MFNet dataset and PST900 dataset under './model/'.
   
   **Performace on MFNet dataset**
   
   <div align=center>
   <img src="https://github.com/MathLee/LASNet/blob/main/Images/MFNet.png">
   </div>
   
   **Performace on PST900 dataset**
   
   <div align=center>
   <img src="https://github.com/MathLee/LASNet/blob/main/Images/PST900.png">
   </div>
 

# Training
1. Download [MFNet dataset](https://pan.baidu.com/s/1NHGazP7pwgEM47SP_ljJPg) (code: 3b9o) or [PST900 dataset](https://pan.baidu.com/s/13xgwFfUbu8zNvkwJq2Ggug) (code: mp2h).
2. Run train_base_vit.py (default to MFNet Dataset).

Note: our main model is under './toolbox/models/LASNet_vit.py'


# Pre-trained model and testing
1. Download the following pre-trained model and put it under './model/'. [model_MFNet.pth](https://pan.baidu.com/s/1dWCbTl274nzgdHGOsJkK_Q) (code: 5th1)   [model_PST900.pth](https://pan.baidu.com/s/1zQif2_8LTG5R7aabQOXjrA) (code: okdq)

2. Rename the name of the pre-trained model to 'model.pth', and then run test_LASNet.py (default to MFNet Dataset).
  
  
# Citation
       
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
