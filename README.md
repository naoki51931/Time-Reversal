# Time-Reversal
Unofficial implementation of "Explorative Inbetweening of Time and Space"

## Setup
```
git clone git@github.com:YingHuan-Chen/Time-Reversal.git && cd Time-Reversal
conda create -n time-reversal python=3.8 -y
conda activate time-reversal
pip install diffusers transformers accelerate opencv-python configargparse
```

## Run
```
python main.py --datadir ./data/lineart --outputdir ./results --num_frames 12 --prompt "anime girl, clean lineart, full body, dynamic pose" --negative_prompt "low quality, blurry" --controlnet_strength 1.0 --easing ease-in-out --contrast 1.2 --sharpness 1.1
```

## Citation
```
@article{feng2024explorative,
  title={Explorative Inbetweening of Time and Space},
  author={Feng, Haiwen and Ding, Zheng and Xia, Zhihao and Niklaus, Simon and Abrevaya, Victoria and Black, Michael J and Zhang, Xuaner},
  journal={arXiv preprint arXiv:2403.14611},
  year={2024}
}
```
