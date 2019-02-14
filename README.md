## Stacked Dense U-Nets for Face Alignment

The Mxnet implementation of Jia Guo's work ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936). 

Some popular heatmap based approaches like stacked hourglass are also provided in this repository.  

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1kdquiIGTlK7l26SPWO_cmw)

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1VjFWm6eEtIqGKk92GE2rgw)

Pre-trained models will come soon.

The code is adapted based on an intial fork from the [insightface](https://github.com/deepinsight/insightface) repository.

## Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

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
@article{Guo2018Stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  year={2018},
}
```

