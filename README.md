# m2v-diffusers

## 环境

### 镜像记录

- 2024.05.24: `m2v-nv:m2v_nv_torch221_cu12_ema_0524`

## 训练

### 单机单卡运行

指定`CONFIG_NAME`和`EXP_NAME`

```shell
python main.py CONFIG_NAME --experiment-name EXP_NAME
```

### 多机多卡训练

```shell
kai_launch kaimm.sh CONFIG_NAME EXP_NAME
```

可以修改参数，例如随机种子
```shell
kai_launch kaimm.sh CONFIG_NAME EXP_NAME --seed 42
```

查看参数设置
```shell
kai_launch kaimm.sh CONFIG_NAME EXP_NAME --help
```



新clone项目需要手动新建`src/configs/train_configs/personal_configs.py`，其内容如下：

```python
import copy
from dataclasses import replace
from typing import Dict

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...pipelines import *


personal_configs: Dict[str, TrainerConfig] = {}
```



## 推理

1B_config: /m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/config.yml

1B_ckpt: /m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/mvb_1b_f77_distill_ema_merged.ckpt

test_csv: /m2v_intern/yuanziyang/share/csv/test_prompt_0527_recaption.csv

timestep_shift 需要调成 5

negative_prompt 建议使用如下的 prompt


多卡运行:
```
bash scripts/dist_run.sh \
        python scripts/m2v_dist_infer.py \
        /m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/config.yml \
    --data.path /m2v_intern/yuanziyang/share/csv/test_prompt_0527_recaption.csv \
    --data.caption_column caption \
    --data.num_samples 225 \
    --data.batch_size 1 \
    --data.cache_dir None \
    --test_dir test_results \
    --transformer_ckpt_path /m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/mvb_1b_f77_distill_ema_merged.ckpt \
    --negative_prompt "animation, 2d animation, 3d animation, Anime, Cartoon, blurry, deformed, disfigured, low quality, text, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted" \
    --width 672 \
    --height 384 \
    --fps 15 \
    --num_frames 77 \
    --guidance_scale 12.5 \
    --seed 42 \
    --num_inference_steps 50 \
    --timestep_shift 5.0



```
单卡运行:

把 scripts/m2v_dist_infer.py 里面的 USE_DIST 设置成 False, 直接 python scripts/m2v_dist_infer.py ......  运行



## Suggestion


- 开发
    - 不进行hard coding
    - 不大量copy外部代码（多个文件夹等），尽量使用 submodule 或 集成+二次开发 的方式加入仓库
    - 考虑通用性和可扩展性，尽量减少反复造轮子
    - 增加必要注释，以提升可读性
    - 遵循[Y-tech代码规范](https://docs.corp.kuaishou.com/d/home/fcABVEmP8rlD8Nphet--WksuD)
    - 公司[git文档](https://docs.corp.kuaishou.com/d/home/fcAALLYHcERNsYhaCvy6qUq7y#section=h.gt04g8u09a09)
- 上传
    - 不上传实验用数据、ckpt、代码，可在gitignore中去除对非主要文件的跟踪
    - 删除不必要注释
    - 保证主分支整洁 和 整体仓库轻量化
- 提交
    - 及时提交重要更新
    - 提交方负责单元测试，和后续debug
    - 项目主R进行review和mege，提交方按需进行讲解，并拉齐相关同学认知
