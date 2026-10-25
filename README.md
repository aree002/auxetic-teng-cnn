# auxetic-teng-cnn

A 1D convolutional neural network that recognises grasped objects from triboelectric nanogenerator (TENG) signals. It reproduces the deep-learning part of Wang et al., *Bioinspired Auxetic Metastructures Enable Biomechanically Adaptive, Machine Learning-Enhanced Self-Powered Sensing with Ultrahigh Efficiency*, Nano-Micro Letters 18:286 (2026).

Part of the ME F366 Lab Oriented Project *Study and Analysis of Triboelectric Energy Harvester using Bio-Inspired Auxetic Structures* (BITS Pilani, Hyderabad Campus).

## What's here

| File | Purpose |
| --- | --- |
| `model.py` | The paper's 1D-CNN (Note S2, Tables S4–S5): Conv 6 → 12 → 60 filters (kernel 5), FC 60, softmax over 6 classes |
| `teng_signals.py` | Generates synthetic grasp signals from a COMSOL electrostatic model of the TENG |
| `train.py` | Builds the dataset, trains the CNN, compares with SVM and random forest, saves figures and metrics |
| `check_load_sweep.py` | Cross-check of the TENG load-resistance circuit model |
| `data/` | COMSOL input table and the generated datasets |
| `results/` | Figures and metrics from the runs reported below |

## Dataset

The paper's recorded signals are not published, so the data here is **synthetic**. Each signal is the voltage across a 100 MΩ load, computed from the TENG circuit equation `R dQ/dt = V_oc(x) − Q/C(x)`. `V_oc(x)` and `C(x)` come from a COMSOL model of a 5 × 5 cm FEP / CA-PEI device (`data/comsol_fe_voc_c_vs_gap.csv`).

- **Classes:** six grasps modelled on Fig. 6j of the paper: hand light touch, hand fist, hand orange, gripper empty, gripper orange, gripper apple.
- **Per class:** 300 signals of 1024 samples (2.048 s at 500 Hz), 1,800 signals in total.
- **Realism:** noise, 50 Hz hum and baseline drift are added.
- **Two versions:**
  - `data/synthetic_grasp_auxetic.npz`: auxetic device, full contact, low noise.
  - `data/synthetic_grasp_non_auxetic.npz`: non-auxetic device, 40% less contact charge and 2.5× the noise. These levels are assumptions, not measurements.

```python
import numpy as np
d = np.load("data/synthetic_grasp_auxetic.npz")
X, y = d["X"], d["y"]   # X: (1800, 1024) voltage in V, y: class 0–5
```

## Results (auxetic signals, 360-signal test set)

| Model | Accuracy | AUC |
| --- | --- | --- |
| 1D-CNN | 90.3% | 0.994 |
| SVM (RBF) | 86.9% | 0.978 |
| Random forest | 72.2% | 0.925 |

The paper reports 98.5% for the CNN on its own recorded data, so the numbers are not directly comparable. The CNN overfits with the paper's training recipe, which has no regularisation.

![Confusion matrix](results/auxetic/fig_confusion_cnn.png)

## Run it

```bash
pip install -r requirements.txt
python train.py --device auxetic        # or --device non_auxetic
```

`train.py` regenerates the dataset (seed 42), trains for 100 epochs on CPU (about 4 minutes), and writes figures and `metrics.json` to `results/<device>/`.

The random forest is built from scikit-learn decision trees, with bootstrap sampling and √n features per split. This is because the compiled `sklearn.ensemble` module was blocked on the development machine; the algorithm is the standard one.
