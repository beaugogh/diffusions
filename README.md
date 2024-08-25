# diffusions

## Installation
system prerequisites:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

python environment:
```
cd path/to/diffusions
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Note: LivePortrait requirements include transformers, exlude that!


## Nvidia issues

Nvidia issue after suspension
```
sudo apt install nvidia-modprobe

sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

