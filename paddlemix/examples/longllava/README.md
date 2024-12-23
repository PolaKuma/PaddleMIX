# LongLLaVA

## 1. 模型介绍

[LongLLaVA](https://arxiv.org/pdf/2409.02889) 是基于大规模语言模型的混合视觉语言模型，专注于增强长上下文处理能力，支持视频理解、高分辨率图像处理等多模态任务，同时在高效性和有效性之间实现了平衡。

<p align="center">
  <img src="https://github.com/FreedomIntelligence/LongLLaVA/blob/main/assets/singleGPU.png" align="middle" width = "600" />
</p>

注：图片引用自[LongLLaVA](https://github.com/FreedomIntelligence/LongLLaVA).


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| FreedomIntelligence/LongLLaVA-9B  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("FreedomIntelligence/LongLLaVA-9B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

版本要求：paddlenlp>=3.0.0b2

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本，Python最低版本要求3.8。


## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 推理
```bash
python paddlemix/examples/longllava/cli.py \
    --model_dir "FreedomIntelligence/LongLLaVA-9B"
```
可配置参数说明：
  * `model_dir`: 指定LongLLaVA的模型名字或权重路径，也可换成如'FreedomIntelligence/LongLLaVA-9B'


### 参考文献
```BibTeX
@misc{wang2024longllavascalingmultimodalllms,
      title={LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architecture}, 
      author={Xidong Wang and Dingjie Song and Shunian Chen and Chen Zhang and Benyou Wang},
      year={2024},
      eprint={2409.02889},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.02889}, 
}
```