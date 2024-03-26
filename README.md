
# NRMM Detection in Satellite Image using YOLOv8

## Project Description

A comprehensive framework that employs YOLOv8, a state-of- the-art object detection algorithm. The methodology involves an initial data preprocessing and augmentation stage to enhance the quality and quantity of our dataset, followed by modification experiments on the architecture to fine-tune its detection capabilities. 

For the performance evaluation, we primarily utilize the metric of mean Average Precision at an Intersection over Union (IoU) threshold of 0.5 (mAP50). This is complemented by additional evaluation methods such as confusion matrices and precision-recall curves to provide a comprehensive assessment of the model’s performance and its limitations.

The resulting model achieves a mAP50 of 20.6% across 14 NRMM-related classes, with some categories like ‘Cement Mixer’ reaching an mAP50 of over 66%. This not only demonstrates the model’s capability in NRMM detection but also indicates its effectiveness in identifying specific classes of machinery.

## Project Documentation

### 1. Dataset Information:
- NRMM-10: This dataset has been preprocessed and augmented. It stands out as the most feasible and optimal dataset after various augmentation experiments. It's the primary dataset used for subsequent model modification experiments.

### 2. Notebook Descriptions:
#### Preprocessing & Augmentations:

Code and Models training outputs from preprocessing and augmentations experiments are saved in this file.

##### 1.YOLOv8_init.ipynb:
- Initial tryout and setup of the YOLOv8 Model.
- Testing with pre-trained COCO dataset YOLOv8.
- Training YOLOv8 on our xView dataset with a one-label task: NRMM and background.
- Conversion of the geojson labeling file into YOLO pytorch txt format.

##### 2. YOLOv8_m2(sourcecode).ipynb:
- Instead of importing the ultralytics packages online, this notebook clones the source code from GitHub.
- Preparation for further experiments.

##### 3. YOLOv8_v2.ipynb:
- Training the YOLOv8 base model using data with the following preprocessing and augmentation:
    - Resize: Stretch to 640x640
    - Tile: 8 rows x 8 columns
    - Modify Classes: 0 remapped, 44 dropped (Only contains NRMM related classes and "Construction site")
    - Filter Null: At least 50% of images must contain annotations. o 90° Rotate: Clockwise, Counter-Clockwise, Upside Down

##### 4. YOLOv8_v3.ipynb:
- Uses images from the relabeling dataset with the following preprocessing and augmentation:
- (Same as YOLOv8_v2.ipynb but with additional steps)
    - Outputs per training example: 4
    - Flip: Horizontal, Vertical
    - Grayscale: Apply to 50% of images

##### 5. YOLOv8_v4.ipynb:
- Uses images from the relabeling dataset.
- Preprocessing and augmentation steps are the same as YOLOv8_v3.ipynb.
- Output Saved at: YOLOv8_v4

##### 6. YOLOv8_v5.ipynb: Exp.5
- Uses images from the relabeling dataset with the following preprocessing and
augmentation:
    - Resize: Stretch to 2408 x 2408
    - (Other steps same as YOLOv8_v3.ipynb)
    - Output Saved at: YOLOv8_v5

##### 7. YOLOv8_v6.ipynb:
- Uses images from the relabeling dataset.
- Preprocessing and augmentation steps are the same as YOLOv8_v3.ipynb, excluding the resize step.
- Output Saved at: YOLOv8_v6

##### 8. YOLOv8_dfl.ipynb: Exp.6
- Uses images from the relabeling dataset.
- Preprocessing and augmentation steps are the same as YOLOv8_v5.ipynb.
- dfl=2

#### Downsampling Expenriments:
##### 1. YOLOv8_m1(down).ipynb:
This notebook utilizes the default configuration to train the YOLOv8 model with specific modifications. The primary objective is to reduce the downsampling factor.

Modifications:
- A. Change in the First Convolutional Layer: Exp.8
    - The stride of the first convolutional layer is adjusted to 1.
    - This alteration decreases the downsampling factor from 32 to 16. 
    - Output Saved at: YOLOv8_down1
- B. Modification in the p2 Layer: Exp.9
    - The downsampling factor is adjusted from 32 to 16. o Output Saved at: YOLOv8_down22
   
- C. Modification in the p3 Layer: Exp.10
    - The downsampling factor is adjusted from 32 to 16. o Output Saved at: YOLOv8_down3


#### Upsampling Experiments:
In this section, we explore experiments related to enhancing the upsampling processes.

##### 1. YOLOv8_m1(up).ipynb:

Modifications:
- Upsampled Raw Data: Exp.11
    - The raw data is upsampled by a factor of 2. o Output Saved at: YOLOv8_up1
- Using YOLOv8-p2 Model: Exp.12
    - This is the official model released by Ultralytics.
    - It implements the p2 detection head, optimized for detecting smaller objects.
    - Output Saved at: YOLOv8_p2

- Combining Upsample & p2 Model: Exp.13
    - This experiment combines the methodologies of the first and second experiments.
    - Output Saved at: YOLOv8_p2&up1

#### IOU Loss Function Experiments:
This section details experiments conducted with various IOU Loss functions.

##### 1. YOLOv8_m2(WiseIOU).ipynb Exp.14 https://arxiv.org/abs/2301.10051
This notebook utilizes the source code to experiment with WISE IOU.
Experiments & Outputs:
- WiseIOU Version 3:
    - Output Saved at: YOLOv8_WiseIOU
- WiseIOU Version 2:
    - Output Saved at: YOLOv8_WiseIOUv2
- WiseIOU Version 1:
    - Output Saved at: YOLOv8_WiseIOUv1
- Source code: https://github.com/sophuang/ultralytics/tree/WiseIoU
      
##### 2. Notebook: YOLOv8_m2(DWIOU).ipynb Exp.15 https://arxiv.org/abs/2110.13389
This notebook employs the source code to experiment with Distance Wassertain IOU.
- Output Saved at: YOLOv8_DWIOU
- Source code: https://github.com/sophuang/ultralytics/tree/DWIOU

#### ODC Convolutional Layer Experiments:
This section details experiments conducted with the addition of a novel convolutional layer, ODC.
##### 1. YOLOv8_m3(ODConv).ipynb
https://openreview.net/forum?id=DmpCfq6Mg39
- Experiment: ODConv after p1
    - Output: YOLOv8_ODCp1
- Experiment: ODConv after p2
    - Output Notebook: YOLOv8_ODCp2
- Experiment: ODCNext
    - Output Saved at: YOLOv8_ODCNext
- Source code: https://github.com/sophuang/ultralytics/tree/ODConv
