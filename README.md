# 🧠 Surgical Phase Recognition with VideoMAE + MS-TCN

This project performs **surgical phase recognition** on the **Cholec80** dataset using a two-stage pipeline:
1. **Pretrained VideoMAE** for extracting clip-level features.
2. **MS-TCN** for temporal modeling and frame-wise phase prediction.

---

## 📦 Installation

### 1. Clone the repository
```bash
- git clone https://github.com/your-username/surgical-phase-recognition.git
- cd surgical-phase-recognition
```
```bash
.
├── extract_features.py      # Uses VideoMAE to extract clip features
├── train_mstcn.py           # Trains MS-TCN using features
├── predict.py               # Runs full inference and predicts phases
├── model/
│   ├── videomae/            # Pretrained VideoMAE model
│   └── mstcn.py             # MS-TCN model definition
├── data/
│   └── cholec80/            # Cholec80 dataset location
│       └── videos/
├── features/                # Saved .pkl feature files
├── checkpoints/             # Trained MS-TCN models
└── requirements.txt
```

### Project setup : ``` pip install -r requirements.txt ```

### Step 1: Pretrain VideoMAE : ```python pretraining.py```
### Step 2: Extract Features : ```python feature_extraction.py```
### Step 3: Train MSTCN : ```python train_mstcn.py```

### Inferencing : ```Use the codes in ./inferencing```

Author : Debashis Baidya, TU dresden, Dresden, Germany