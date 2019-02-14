The Mxnet implementation of Jia Guo's work ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936).

Some popular heatmap based approaches like stacked hourglass are also provided.  You can define different loss-type/network structure/dataset in ``config.py``(from ``sample_config.py``).

For example, by default, you can train this approach by ``train.py --network sdu`` or train hourglass network by ``train.py --network hourglass``.

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1kdquiIGTlK7l26SPWO_cmw)

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1VjFWm6eEtIqGKk92GE2rgw)

Pre-trained models will come soon.
