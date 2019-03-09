## Stacked Dense U-Nets for Face Alignment

The Mxnet implementation of the most recent state-of-the-art 2D and 3D face alignment method ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936). This proposed stacked dense U-Nets (SDU-Net) with dual transformers can get the normalised mean error (NME) of 6.73% and 5.55% respectively on IBUG and COFW datasets for 2D face alignment. For 3D face alignment, this method further decreases the NME by 5.8% with former state-of-the-art method (HPM) proposed by Bulat et al on AFLW2000-3D.

3D alignment results on the AFLW2000-3D dataset:

| Method    | ESR   |  RCPR   |  SDM   | 3DDFA  |  HPM  | SDU-Net |
|  :------: | :----:  |  :----:  |  :----:   |  :----:  |  :----:   |  :----:  |
|   NME     |   7.99   |   7.80   |   6.12   |   4.94   |   3.26   |   3.07   |

Some popular heatmap based approaches like stacked hourglass are also provided in this repository.  

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1idA68ga8ey-R9TGSwWO62A) and [googledrive](https://drive.google.com/open?id=1XyZ5yFm-MGNlUiGG0dYRHRTdRiS33zPb).

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1EbSx_j_GoNJqLwZyuclBAQ) and [googledrive](https://drive.google.com/open?id=1i-gUFJhtiZP3uCmNbhLCzd4C4fb-Ljhk).

2D pre-trained model will be added soon.

| Network    | IBUG  |  COFW  |
|  :------:   | :----:  |  :----:  |
| SDU-Net Official    |  6.73%  |  5.55%  |
| SDU-Net Ours        |  **–**  |  **–**  |

3D pre-trained model will be added soon.

| Network    | AFLW2000-3D  |
|  :------:   | :----:  | 
| SDU-Net Official    |  3.07%  |
| SDU-Net Ours        |  **–**  |


## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)

## Training

1.  Prepare the environment.

2.  Clone the repository.

3.  Download the training/validation dataset and unzip it to your project directory.
    
3.  You can define different loss-type/network structure/dataset in ``config.py``(from ``sample_config.py``).
    
4.  You can use ``train.py --network sdu`` to train SDU-net or ``train.py --network hourglass`` to train stacked hourglass network.

## Testing

  -  Download the pre-trained model and place it in *`./model_2d/`* or *`./model_3d/`*.

  -  You can use `python test.py` to test this alignment method.
  
## Results

![2D Alignment Results](https://raw.githubusercontent.com/deepinx/sdu-face-alignment/master/sample-images/landmark_test.png)

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
```

## Acknowledgment

The code is adapted based on an intial fork from the [insightface](https://github.com/deepinsight/insightface) repository.

