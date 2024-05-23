<p align="center">
    <img src="./assets/logo.png" width="250"/>
</p>
<h2 align="center"> Pandora: Towards General World Model with Natural Language Actions and Video States</h2>

We introduce Pandora, a step towards a General World Model (GWM) that:
1. Simulates world states by generating videos across any domains
2. Allows any-time control with actions expressed in natural language

**Please refer to [world-model.ai](world-model.ai) for results.**

[[Website]](https://world-model.maitrix.org/)
[[Paper]](https://world-model.maitrix.org/assets/pandora.pdf)
[[Model]](https://huggingface.co/maitrix-org/Pandora)
[[Gallery]](https://world-model.maitrix.org/gallery.html)

## News
- __[2024/05/23]__ Release the model and inference code.
- __[2024/05/23]__ Launch the project page and release the paper.

## Setup
```shell
conda create -n pandora python=3.12.3 nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda activate pandora
pip install torch torchvision torchaudio
bash build_envs.sh  # other pip requirements
```

## Inference
### Gradio Demo
1. Download the model checkpoint from [Hugging Face](https://huggingface.co/maitrix-org/Pandora)
2. Run the commands on your terminal
```shell
CUDA_VISIBLE_DEVICES={cuda_id} python gradio_app.py  --ckpt_path {path_to_ckpt}
```

Then you can interact with the model through gradio interface.

## Citation
```bib
@article{xiang2024pandora,
  title={Pandora: Towards General World Model with Natural Language Actions and Video States},
  author={Jiannan Xiang and Guangyi Liu and Yi Gu and Qiyue Gao and Yuting Ning and Yuheng Zha and Zeyu Feng and Tianhua Tao and Shibo Hao and Yemin Shi and Zhengzhong Liu and Eric P. Xing and Zhiting Hu},
  year={2024}
}
```
