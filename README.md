# diffusions

## Installation
#### system prerequisites:
```bash
sudo apt install ffmpeg
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### python environment:
```
cd path/to/diffusions
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Note: LivePortrait requirements include transformers, exclude that!


## Model Weights

#### download LivePortrait model (see [here](https://huggingface.co/KwaiVGI/LivePortrait))
Note that the downloaded models need to be put into `./LivePortrait/pretrained_weights`, the default model directory can be modified in `./LivePortrait/src/config/inference_config.py`, but not necessary. For some reason, the huggingface repo does not contain all the model files, you can just put the downloaed `liveportrait` models, the script will automatically download the rest, more complete version of models. 
```
huggingface-cli download KwaiVGI/LivePortrait --local-dir ./models/live_portrait_hf --exclude "*.git*" "README.md" "docs"
```
#### download DensePose model (see [here](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/TOOL_APPLY_NET.md))
```
wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
```

## Quickfixes

#### Nvidia issue after suspension
```
sudo apt install nvidia-modprobe

sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

