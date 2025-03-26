# ASIP
Code for our paper ASIP: **Accurate and Steady Inertial Pose Estimation through Sequence Structure Learning and Modulation**. 

## Environment Setup

### Install dependencies
We use ```python 3.7.6```. You should install the newest ```pytorch chumpy vctoolkit open3d```.

### Prepare SMPL body model
Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click SMPL for Python and download the version 1.0.0 for Python 2.7 (10 shape PCs). Then unzip it.

### Prepare pre-trained model
You can download our pre-trained model from [here](https://pan.baidu.com/s/1BxD0FC19Lxy_bf3aOeNaLw?pwd=lhu7).

## Preprocessing Data
- Following [Transpose](https://github.com/Xinyu-Yi/TransPose), we preprocess the AMASS, DIP and TotalCapture datasets.
- Following [DynaIP](https://github.com/dx118/dynaip), we preprocess the AnDy and CIP datasets.
- The raw data of these datasets can be downloaded from [AMASS](https://amass.is.tue.mpg.de/), [DIP](https://dip.is.tue.mpg.de/), [TotalCapture](https://cvssp.org/data/totalcapture/), [AnDy](https://zenodo.org/records/3254403) and [CIP](https://zenodo.org/records/5801928).

## Running the Evaluation
```python common/eval_all.py```
  
## Acknowledgment
We thank that these two repositories [Transpose](https://github.com/Xinyu-Yi/TransPose) and [DynaIP](https://github.com/dx118/dynaip) have provided many useful code. 
## Citation

If you find the project helpful, please consider citing us:
```bibtext
@inproceedings{asip,
  title={Accurate and Steady Inertial Pose Estimation through Sequence Structure Learning and Modulation},
  author={Wu, Yinghao and Wang, Chaoran and Yin, Lu and Guo, Shihui and Qin, Yipeng},
  booktitle={NeurIPS},
  year={2024}
}
```
