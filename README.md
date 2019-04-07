## Robust 2D and 3D Face Alignment Implemented in MXNet

This repository contains several heatmap based approaches like stacked Hourglass and stacked Scale Aggregation Topology (SAT) for robust 2D and 3D face alignment. Some popular blocks such as bottleneck residual block, inception residual block, parallel and multi-scale (HPM) residual block and channel aggregation block (CAB) are also provided for building the topology of the deep face alignment network. All the codes in this repo are implemented in Python and MXNet.

The models for 2D face alignment are verified on IBUG, COFW and 300W test datasets by the normalised mean error (NME) respectively. For 3D face alignment, the 3D pre-trained models are compared on AFLW2000-3D with the most recent state-of-the-art methods.

The training/validation dataset and testset are in below table:

| Data | Download Link | Description |
|:-:|:-:|:-:|
| train_testset2d.zip | [BaiduCloud](https://pan.baidu.com/s/1EbSx_j_GoNJqLwZyuclBAQ) or [GoogleDrive](https://drive.google.com/open?id=1XyZ5yFm-MGNlUiGG0dYRHRTdRiS33zPb), 490M | 2D training/validation dataset and IBUG, COFW, 300W testset |
| train_testset3d.zip | [BaiduCloud](https://pan.baidu.com/s/1idA68ga8ey-R9TGSwWO62A) or [GoogleDrive](https://drive.google.com/open?id=1i-gUFJhtiZP3uCmNbhLCzd4C4fb-Ljhk), 1.54G | 3D training/validation dataset and AFLW2000-3D testset |


The performances of 2D pre-trained models are shown below. Accuracy is reported as the Normalised Mean Error (NME). To facilitate comparison with other methods on these datasets, we give mean error normalised by the eye centre distance. Each training model is denoted by Topology^StackBlock (d = DownSampling Steps) - BlockType - OtherParameters.

| Model | Model Size | IBUG  | COFW  | 300W  | Download Link |
|:-:|:-:|:-:| :-: | :-: | :-: |
| *Hourglass2(d=4)-Resnet* | 26MB |  7.719  |  6.776  |  6.482  | [BaiduCloud](https://pan.baidu.com/s/1xGXiykKrRyGKPXMXDRsMZw) or [GoogleDrive](https://drive.google.com/open?id=1YPfF3t4J33Zj5goIZBk15TsxbqHB90rR) |
| *Hourglass2(d=3)-HPM*    | 38MB |  7.249  |  6.378  |  6.049  | [BaiduCloud](https://pan.baidu.com/s/1qOD-qthPqScsX913EMwKag) or [GoogleDrive](https://drive.google.com/open?id=1-rDuuzxw9civqz9wTtklYqT6k3utr6Gc) |
| *Hourglass2(d=4)-CAB*    | 46MB |  7.168  |  6.123  |  5.684  | [BaiduCloud](https://pan.baidu.com/s/1sSfnxf9_myl7NS7QEddOfQ) or [GoogleDrive](https://drive.google.com/open?id=1o--WwpHoRw2W5bScan6t16vEKS53WBBm) |
| *SAT2(d=3)-CAB*          | 40MB |  7.052  |  5.999  |  5.618  | [BaiduCloud](https://pan.baidu.com/s/1YQKaUwpBq1IWz8vawj6HWA) or [GoogleDrive](https://drive.google.com/open?id=1n-Nd4rdik-IWqIzgIEdssDKvZ7SwuOff) |
| *Hourglass2(d=3)-CAB*    | 37MB |**6.974**|**5.983**|**5.647**| [BaiduCloud](https://pan.baidu.com/s/1BysgX7X2p1g8X8nS01gFlA) or [GoogleDrive](https://drive.google.com/open?id=1AbTGhIBzUUINTH2GNL05tSWvOHnclRr4) |


The performances of 3D pre-trained models are shown below. Accuracy is reported as the Normalised Mean Error (NME). The mean error is normalised by the square root of the ground truth bounding box size.

| Model | Model Size | AFLW2000-3D  | Download Link |
| :-: | :-: | :-: | :-: |
| *SAT2(d=3)-CAB-3D* | 40MB |  3.072  | [BaiduCloud](https://pan.baidu.com/s/1kFO-kTk-ozZ2m494xoNbqw) or [GoogleDrive](https://drive.google.com/open?id=1lxHbWZa_l457oFX4tgSpFaubDKU7LNuq) |
| *Hourglass2(d=3)-CAB-3D* | 37MB |  **3.005**  | [BaiduCloud](https://pan.baidu.com/s/1O2tBppPu6cOPLgqLmMvowA) or [GoogleDrive](https://drive.google.com/open?id=17rk_MiI_7CLfKqV8dsXvRyd56sfBcqMD) |


Note: More pre-trained models will be added soon.

## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.

## Training

  -  Download the training/validation dataset and unzip it to your project directory.
    
  -  You can define different loss-type/network topology/dataset in ``config.py``(from ``sample_config.py``).
    
  -  You can use ``CUDA_VISIBLE_DEVICES='0' train.py --network satnet`` to train stacked Scale Aggregation Topology (SAT) networks or ``CUDA_VISIBLE_DEVICES='0' train.py --network hourglass`` to train stacked Hourglass models. Instead, you can also edit  _`train.sh`_  and run  _`sh train.sh`_  to train your models.

## Testing

  -  Download the ESSH model from [BaiduCloud](https://pan.baidu.com/s/1sghM7w1nN3j8-UHfBHo6rA) or [GoogleDrive](https://drive.google.com/open?id=1eX_i0iZxZTMyJ4QccYd2F4x60GbZqQQJ) and place it in *`./essh-model/`*.

  -  Download the pre-trained model and place it in *`./models/`*.

  -  You can use `python test.py` to test your models for 2D and 3D face alignment.
  
## Results

Results of 2D face alignment (inferenced from model *Hourglass2(d=3)-CAB*) are shown below.

![2D Alignment Results](https://raw.githubusercontent.com/deepinx/sdu-face-alignment/master/sample-images/landmark_test_2d.png)

Results on ALFW-2000 dataset (inferenced from model *Hourglass2(d=3)-CAB-3D*) are shown below.

![3D Alignment Results](https://raw.githubusercontent.com/deepinx/sdu-face-alignment/master/sample-images/landmark_test_3d.png)

## License

MIT LICENSE


## Reference

```
@article{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1812.01936},
  year={2018}
}

@inproceedings{Deng2018Cascade,
  title={Cascade Multi-View Hourglass Model for Robust 3D Face Alignment},
  author={Deng, Jiankang and Zhou, Yuxiang and Cheng, Shiyang and Zaferiou, Stefanos},
  booktitle={2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018)},
  pages={399-403},
  year={2018},
}

@article{Bulat2018Hierarchical,
  title={Hierarchical binary CNNs for landmark localization with limited resources},
  author={Bulat, Adrian and Tzimiropoulos, Yorgos},
  journal={IEEE Transactions on Pattern Analysis & Machine Intelligence},
  year={2018},
}

@inproceedings{Jing2017Stacked,
  title={Stacked Hourglass Network for Robust Facial Landmark Localisation},
  author={Jing, Yang and Liu, Qingshan and Zhang, Kaihua and Jing, Yang and Liu, Qingshan and Zhang, Kaihua and Jing, Yang and Liu, Qingshan and Zhang, Kaihua},
  booktitle={IEEE Conference on Computer Vision & Pattern Recognition Workshops},
  year={2017},
}
```

## Acknowledgment

The code is adapted based on an intial fork from the [insightface](https://github.com/deepinsight/insightface) repository.

