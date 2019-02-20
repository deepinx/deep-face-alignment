## Stacked Dense U-Nets for Face Alignment

The Mxnet implementation of Jia Guo's work ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936). 

Some popular heatmap based approaches like stacked hourglass are also provided in this repository.  

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1idA68ga8ey-R9TGSwWO62A).

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1EbSx_j_GoNJqLwZyuclBAQ).

Pre-trained models will come soon.


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

  You can use `python test.py` to test this alignment method.

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

