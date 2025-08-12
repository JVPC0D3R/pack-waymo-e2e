# Pack WOD End to End Driving

![res/traj_no_front_view.png]

### Installation

```bash
conda create -n packe2e python=3.9
conda activate packe2e
git clone https://github.com/JVPC0D3R/pack-waymo-e2e.git
cd pack-waymo-e2e
pip install -e .
```

### Packing

```bash
cd src/
python convert.py --mode train --input path/to/waymo --output path/to/h5
```