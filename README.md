# Multi_threading-Network
Python code for MTDCNet, There is a low parameter 3D Network for the purpose of tumor segmentation based on pytorch.

Dependency: 
- Python 3.6.0+
- Torch 0.4.0+

Sofware to Visualize: 
- ITK-Snap (available at http://www.itksnap.org/pmwiki/pmwiki.php).

Dataset: 
- The BraTS 2018 train/validation dataset can be downloaded from https://ipp.cbica.upenn.edu/, please register first.

Preprocessing data
- "python pre.py", the dataset shoule be placed at ./data/ and a textfile with the nii file path.
 

How to segment the tumor(for test/val): 
- "python segment.py"
![image](https://user-images.githubusercontent.com/63543796/194700002-48d5da13-5e12-4f88-bead-4cd3ef72cc19.png)

(A) Flair, T1, T1ce and T2, (B) Ground Truth, (C) Baseline and (D) Our-Net.

How to vaildation: 
- Upload the segment file to https://ipp.cbica.upenn.edu/#BraTS18eval_validationPhase, please register first.
