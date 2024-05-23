# Pandora: Towards General World Model with Natural Language Actions and Video States

[[Project]](https://world-model.maitrix.org/)
[[Paper]](https://world-model.maitrix.org/assets/pandora.pdf)
[[Model]](https://huggingface.co/maitrix-org/Pandora)
[[Gallery]](https://world-model.maitrix.org/gallery.html)

## News
- __[2024/05/23]__ Release the model and inference code.
- __[2024/05/23]__ Launch the project page and release the paper.

## TODO List
* [x] Model weights.
* [x] Inference code.
* [ ] Dataset.
* [ ] Data processing pipeline.
* [ ] Training code.

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
2. Run the commands on your terminal (we only support 
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

## Acknowledgement
* [Dynamicrafter](https://github.com/Doubiiu/DynamiCrafter/tree/main): Animating Open-domain Images with Video Diffusion Priors
  system.
* [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi): Unified Visual Representation Empowers Large Language Models with Image and Video Understanding.
