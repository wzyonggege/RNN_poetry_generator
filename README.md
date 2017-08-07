# RNN_poetry_generator

基于RNN生成古诗

### 环境

- python3.6
- tensorflow 1.2.0

### 使用

- 训练：

<code>python poetry_gen.py --mode train</code>

- 生成：

<code>python poetry_gen.py 或者 python poetry_gen.py --mode sample</code>

- 生成藏头诗：

<code>python poetry_gen.py --mode sample --head 明月别枝惊鹊</code>

> 生成藏头诗 --->  明月别枝惊鹊

> 明年襟宠任，月出画床帘。别有平州伯性悔，枝边折得李桑迷。惊腰每异年三杰，鹊出交钟玉笛频。

### 帮助

<code>python poetry_gen.py --help </code>

<code>
  usage: poetry_gen.py [-h] [--mode MODE] [--head HEAD]

  optional arguments:
    -h, --help   show this help message and exit
    --mode MODE  usage: train or sample, sample is default
    --head HEAD  生成藏头诗
    </code>
