# Master Thesis Codebase – Event Spotting and Action Valuation in Football

This repository contains the code used in the context of my master's thesis, focused on end-to-end spotting of ball actions and action valuation in football videos. The project is structured into three main folders:

---

## 📁 `T-DEED/` — Ball Action Spotting

This folder contains the source code from the official T-DEED GitHub repository, a state-of-the-art end-to-end model for dense event spotting in football videos. It includes all necessary components for feature extraction, temporal modeling, and inference.

- **My personal contribution** is included in the script `inference_test.py`, which is used to run inference on a custom set of frames using a pretrained T-DEED checkpoint.
- All pretrained models and datasets have been removed for space reasons. However, every dependency and required file is thoroughly documented in the thesis report.

---

## 📁 `Valuing/` — Action Valuation Module

This folder includes all the code related to the valuation of football actions using predictive models.

- It contains binary classifiers for estimating the probability of scoring and conceding.
- Includes code for **feature engineering**, **pre- and post-processing**, as well as **evaluation routines**.
- Both the baseline (SPADL + VAEP) and enhanced versions (with complex features) are implemented here.

---

## 📁 `soccernet/` — Baseline & Support Code

This folder extends the baseline code for the SoccerNet challenge and includes various personal modifications and additions:

- I directly integrated my custom code into the necessary modules for improved workflow.
- The `OCR/` subfolder contains the code needed to run **jersey number recognition**, based on my custom implementation combining legibility detection, pose estimation, and PARSeq-based OCR.
- Additional utility scripts and analysis tools are also present throughout the folder to assist with external evaluations and data inspection.

---

## Notes

- **Datasets and pretrained models have been removed** from the repository to save space.
- Every required resource (datasets, checkpoints, configuration files) is clearly described and documented in the thesis report.
