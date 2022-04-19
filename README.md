# Multi_threading-Network
Python code for MTDCNet: A 3D multi-threading dilated convolutional network for brain tumor automatic segmentation
There is a low parameter 3D Network in Torch for the purpose of tumor segmentation.

![net](https://user-images.githubusercontent.com/63543796/163979942-f94b2690-3715-4d08-bfaa-436f96bbd899.jpg)
Here is the main structure of MTDC-Net.

Dependency
Python 3.6.0+
Torch 0.4.0+

Sofware to Visualize
ITK-Snap(available at http://www.itksnap.org/pmwiki/pmwiki.php).

Dataset
The BraTS 2018 dataset can be downloaded from https://ipp.cbica.upenn.edu/, please register first.

Preprocessing data
"python pre.py", the dataset shoule be placed at /root/data/train.
 
For model training
"python train.py"

Sebment the tumor(for test/val)
"python segment.py"
![tumor](https://user-images.githubusercontent.com/63543796/163986166-72439f92-a208-490b-b6b3-2055b40c1d5f.jpg)
(A) Flair, T1, T1ce and T2, (B) Ground Truth, (C) Baseline and (D) MTDC-Net.

Vaildation
Upload the segment file to https://ipp.cbica.upenn.edu/#BraTS18eval_validationPhase, please register first.

