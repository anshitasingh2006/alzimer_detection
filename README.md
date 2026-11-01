# Alzheimer Detection Using Speech Recognition

A cutting-edge AI project that uses **speech recognition and machine learning** to detect early signs of *Alzheimer’s disease* from voice recordings.

This repository combines speech preprocessing, feature extraction, and predictive modeling to build a non-invasive system for detecting cognitive decline from speech patterns — empowering early diagnosis with modern AI techniques.

---
## About the Project

Alzheimer Detection Using Speech Recognition is an AI-driven system designed to identify early signs of Alzheimer’s disease through voice analysis.

The project leverages speech recognition, natural language processing (NLP), and machine learning techniques to analyze linguistic and acoustic features such as pauses, fluency, pronunciation patterns, and semantic structure.

By detecting subtle speech abnormalities associated with cognitive decline, this system aims to provide a non-invasive, cost-effective, and accessible tool to support early diagnosis and medical research.

This project demonstrates the intersection of:
- Artificial Intelligence
- Healthcare Technology
- Speech Processing
- Cognitive Analysis
---

## Project Overview

Alzheimer’s is a progressive neurological disorder that impacts memory, behavior and communication. Speech carries subtle cues — like pauses, prosody, fluency and language use — that can signal cognitive impairment. By leveraging **AI and speech recognition**, this project aims to classify speech samples into Alzheimer’s vs. non-Alzheimer’s categories using deep learning and traditional ML models.

---

## Features

- **Speech Recognition**
  - Transcribes audio recordings into text using an ASR engine.

- **AI-Driven Detection**
  - Extracts acoustic and linguistic features from speech.
  - Trains ML/DL models to predict early cognitive signs.

- **Visual Results & Predictions**
  - Outputs probability scores and prediction results.
  - (Optional) Visualization of spectrograms/features.

- **Hybrid Deep Learning + Fuzzy Logic System (XAI)**
  - Interpretable Mamdani Fuzzy Inference System integrating deep neural probabilities with 5 acoustic biomarkers.
  - Calibrated membership functions for pause durations, intra-sentence hesitations, phonation rate, jitter, and shimmer.
  - Transparent rule activation tracking, centroid defuzzification (0–100 score), and neural-acoustic discordance detection.
  - Complete architecture, formulas, and evaluation benchmarks documented in [HYBRID_FUZZY_SYSTEM.md](HYBRID_FUZZY_SYSTEM.md).

- Modular backend and frontend support for easy extension.

---

## Repo Structure

```text
alzimer_detection/
│
├── Backend/                       # AI models, speech processing & API logic
│   ├── app.py                     # Main FastAPI application
│   ├── inference.py               # Unified DL + Fuzzy prediction pipeline
│   ├── fuzzy_rules.py             # Empirical membership functions & Mamdani rules
│   ├── fuzzy_logic.py             # Fuzzy inference engine & XAI report generator
│   ├── fuzzy_visualization.py     # Membership and defuzzification plotting
│   ├── evaluate_hybrid_system.py  # 40-audio validation benchmark script
│   ├── test_fuzzy_logic.py        # Automated unit tests
│   ├── models/                    # Trained CNN-LSTM & GRU attention models
│   ├── utils/                     # Feature extraction & audio preprocessing
│   └── requirements.txt           # Backend dependencies
│
├── Frontend/                      # User interface
│   ├── public/                    # Static assets & generated plots
│   ├── src/                       # React frontend source code & Fuzzy UI
│   └── package.json               # Frontend dependencies
│
├── HYBRID_FUZZY_SYSTEM.md         # Full research & engineering specifications
├── README.md                      # Project documentation
└── .gitignore                     # Ignored files
```

