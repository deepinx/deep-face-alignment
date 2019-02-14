## Stacked Dense U-Nets for Face Alignment

The Mxnet implementation of Jia Guo's work ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936).

Some popular heatmap based approaches like stacked hourglass are also provided in this project.  

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1kdquiIGTlK7l26SPWO_cmw)

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1VjFWm6eEtIqGKk92GE2rgw)

Pre-trained models will come soon.

## Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

## Training

1.  Prepare the environment.

2.  Clone the repository.

3.  Download the training/validation dataset and unzip it to your project folder.
    
3.  You can define different loss-type/network structure/dataset in ``config.py``(from ``sample_config.py``).
    
4.  You can use ``train.py --network sdu`` to train SDU-net or ``train.py --network hourglass`` to train stacked hourglass network.

## Testing

  You can use `python test.py` to test this alignment method.



