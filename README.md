# 🧠 Surgical Phase Recognition with VideoMAE

This repository extends the official [VideoMAE repository](https://github.com/MCG-NJU/VideoMAE) for the project: **Towards Temporal Foundation Models for Surgical Phase Recognition in Minimally Invasive Cholecystectomy**.

This project focuses on **surgical phase recognition** using the **Cholec80** dataset with VideoMAE, leveraging a sliding window training approach.

---

## 📚 Resources

- **Inference Notebook**: [VideoMAE Segmentation on Kaggle](https://www.kaggle.com/code/debashisbaidya/videomae-segmentation)
- **Sample Fine-Tuning Setup Notebook**: [Google Colab Notebook](https://colab.research.google.com/drive/1mkltcZKbz4Pf8xLRrB-Oy_BcdxPrf2uT?usp=sharing)
- **Full VideoMAE Cholec80 Fine-Tuning Dataset**: [Google Drive Folder](https://drive.google.com/drive/folders/1ErvZnTF-5Kn9khLFeKo9weoJgNxnFRxx?usp=sharing)

---

## 📦 Directory Structure

```bash
.
├── videomae/                # VideoMAE pre-training and fine-tuning code
├── video_extractor/         # For data preparation

```

### For VideoMAE Training : ``` videomae/ ```

### 1. Video Preparation ```video_extractor.py```
### 2. Pre-train VideoMAE ``` run_mae_pretraining_with_chart.py```
### 3. Fine-tune VideoMAE ```run_class_finetuning_with_chart.py```
### Inferencing : ```run_segmentation_inference.py```

Author : Debashis Baidya, TU dresden, Dresden, Germany