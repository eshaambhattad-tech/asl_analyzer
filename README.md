# ASL Analyzer

Real-time American Sign Language (ASL) alphabet recognition from a webcam feed using MediaPipe hand landmarks and a Random Forest classifier.

![Python](https://img.shields.io/badge/python-3.11-blue) ![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.14-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)

---

## Overview

This is **Phase 1** of a larger vision: an AI signer — a memoji-style avatar that can sign speech it hears in real time. Phase 1 is scoped to static ASL alphabet recognition (A–Y, excluding J and Z which require motion).

**Approach:** Instead of training on raw pixels, MediaPipe Hands extracts 21 3D hand landmarks per frame. A Random Forest classifier runs on a 63-dimensional normalized feature vector — making the model small, fast, lighting-invariant, and trainable on a laptop with no GPU.

**Roadmap:**
- **Phase 1 (current):** Real-time static ASL alphabet recognition from webcam
- **Phase 2 (planned):** English-to-ASL-gloss translation (speech/text in, ASL grammar out)
- **Phase 3 (planned):** Animate a rigged avatar to perform signs in real time

---

## Pipeline

```
landmark_check.py   →   capture.py   →   train.py   →   infer.py
  (sanity check)       (collect data)    (train model)   (live demo)
```

### Scripts

| File | Purpose |
|------|---------|
| `landmark_check.py` | Sanity-check that MediaPipe hand tracking works on your webcam |
| `capture.py` | Labeled data collection — press a letter key to record landmarks for that letter |
| `train.py` | Trains a Random Forest on `data/raw/landmarks.csv`, saves model to `model/` |
| `infer.py` | Real-time inference — webcam → predict → display letter + confidence on screen |

---

## Setup

### 1. Install Python via pyenv

> **Do not use system Python or Homebrew Python on macOS** — both have known issues with MediaPipe on Apple Silicon (missing pip, broken `pyexpat`/`libexpat` linking). pyenv sidesteps all of this.

```bash
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9
```

### 2. Create a virtual environment

```bash
python3 -m venv asl_env
source asl_env/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python numpy pandas scikit-learn mediapipe==0.10.14
```

> **Pin mediapipe to 0.10.14.** Versions 0.10.31+ have a confirmed upstream bug where `mp.solutions` doesn't exist. See [google-ai-edge/mediapipe #6200](https://github.com/google-ai-edge/mediapipe/issues/6200).

### 4. Camera permissions (macOS)

Go to **System Settings → Privacy & Security → Camera** and grant Terminal access.

---

## Usage

Activate the venv first in every new terminal session:

```bash
source asl_env/bin/activate
```

### Step 1 — Verify tracking works

```bash
python3 landmark_check.py
```

You should see your hand with 21 landmark dots drawn on it. Press `q` to quit.

### Step 2 — Collect training data

```bash
python3 capture.py
```

- Press a letter key (`a`–`y`, excluding `j`/`z`) to start recording samples for that letter
- Press the same key again or `Space` to stop
- Aim for ~150 samples per letter with slight variation in angle and distance
- Press `q` to quit and save

Data is saved to `data/raw/landmarks.csv` and accumulates across sessions.

### Step 3 — Train the model

```bash
python3 train.py
```

Prints accuracy, per-letter performance, and a confusion matrix. Saves the model to `model/asl_classifier.pkl`.

### Step 4 — Run live inference

```bash
python3 infer.py
```

Hold your hand up to the webcam. The predicted letter appears large on screen with a confidence percentage. Below 50% confidence shows `?`.

---

## How normalization works

Both `capture.py` and `infer.py` apply the same normalization so training and inference features match:

1. **Translate** — shift all 21 landmarks so the wrist (landmark 0) is at the origin
2. **Scale** — divide by the distance to the farthest landmark, making it scale-invariant to hand size and camera distance
3. **Flatten** — produce a 63-length vector: `x0, y0, z0, x1, y1, z1, ..., x20, y20, z20`

This means the model learns hand *shape*, not hand *position* in the frame.

---

## Letters supported

**A B C D E F G H I K L M N O P Q R S T U V W X Y**

J and Z are excluded in Phase 1 — they involve motion and require temporal/sequence modeling.
