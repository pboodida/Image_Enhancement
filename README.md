# Low-Light Image Enhancement with an Attention-Augmented U-Net

A deep learning model that restores natural brightness, contrast, and detail to images captured in low-light conditions — built on a U-Net backbone enhanced with residual learning, channel/spatial attention (CBAM), and attention-gated skip connections.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Datasets](#2-datasets)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Loss Function & Metrics](#5-loss-function--metrics)
6. [Training Configuration](#6-training-configuration)
7. [Results & Evaluation](#7-results--evaluation)
8. [Project Structure](#8-project-structure)
9. [How to Run](#9-how-to-run)
10. [Citations & References](#10-citations--references)

---

## 1. Overview

Photos taken in poor lighting suffer from low brightness, crushed shadows, color distortion, and noise that simple brightness/contrast adjustments can't recover — the missing detail genuinely isn't recoverable through linear correction alone. This project trains a convolutional neural network to learn the mapping from a low-light image to its well-lit counterpart directly from paired training data, recovering detail and color that naive gamma correction or histogram equalization can't reconstruct.

The core architecture is a **U-Net** — the encoder-decoder design with skip connections originally developed for biomedical image segmentation — adapted here for image-to-image restoration and substantially extended with:

- **Residual blocks** at every encoder/decoder stage, for more stable gradient flow at depth
- **CBAM (Convolutional Block Attention Module)** — channel attention and spatial attention applied inside every residual block, so the network learns *what* features matter and *where* they matter
- **Attention gates** on the skip connections themselves, so the decoder doesn't just copy raw encoder features forward, but learns to weight them based on relevance to the current decoding stage
- **Global residual learning** — the network predicts a correction added back onto the original input, rather than reconstructing the entire image from scratch

## 2. Datasets

The model is trained on a combination of three low-light image pairs sources, all using the same paired structure (a low-light image and its corresponding well-lit ground truth of the same scene):

| Dataset | Description |
|---|---|
| **LOL Dataset** (`our485` / `eval15` splits) | The original LOL ("LOw-Light") paired dataset — 485 training pairs, 15 evaluation pairs, captured directly under real low-light conditions |
| **LOL-v2 Real_captured** | Real-world low-light/normal-light pairs, an extension of the original LOL dataset with additional real captured scenes |
| **LOL-v2 Synthetic** | Synthetically darkened versions of normal-light images, used to expand training diversity beyond what real captures alone provide |

All three sources are combined, shuffled together, and split 90/10 into training and validation sets before training begins.

## 3. Data Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Raw image pairs │     │  Normalize to    │     │  Combine LOL +   │
│  (.png, varying  │ ──▶ │  [0, 1] float32  │ ──▶ │  LOL-v2 Real +   │
│  resolutions)    │     │  range           │     │  LOL-v2 Synthetic│
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                              │
                                                              ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  90/10 train/val │ ◀── │  Shuffle combined│ ◀── │  Sanity check:   │
│  index split     │     │  low/high pairs  │     │ equal pair counts│
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│             EnhancedImageGenerator (per-batch, on-the-fly)       │
│                                                                  │
│   • Resize to 384×384 (bicubic)                                  │
│   • [training only] Random flip (horizontal & vertical)          │
│   • [training only] Random brightness / contrast / saturation /  │
│     hue jitter — applied to the LOW-LIGHT input only, never the  │
│     ground truth, so the model learns robustness to lighting     │
│     variation without corrupting its target                      │
│   • Clip to valid [0, 1] range                                   │
└──────────────────────────────────────────────────────────────────┘
```

A custom `tf.keras.utils.Sequence` subclass (`EnhancedImageGenerator`) handles batching and augmentation, applying color-space augmentation exclusively to the input image — the ground truth is only ever geometrically transformed (flipped/rotated in lockstep with the input), never color-jittered, which keeps the supervision signal clean.

## 4. Model Architecture

The network is an **attention-augmented U-Net** with four downsampling/upsampling stages. Below is the actual structure as implemented:

```
                                   INPUT (384×384×3)
                                          │
                          ┌───────────────┴───────────────┐
                          │   Initial Conv Block (×2)     │
                          │   Conv2D → BatchNorm → ReLU   │
                          └───────────────┬───────────────┘
                                          │ conv1 (skip )─────────────────------─┐
                                ▼ MaxPool                                        │
                          ┌─────────────────────┐                                │
                          │ Residual Block ×1   │ conv2 (skip) ────────-------┐  │
                          │ (filters × 2)       │                             │  │
                          └──────────┬──────────┘                             |  |
                                ▼ MaxPool                                     │  │
                          ┌─────────────────────┐                             │  │
                          │ Residual Block ×2   │ conv3 (skip) ──────------┐  │  │
                          │ (filters × 4)       │                          │  │  │
                          └──────────┬──────────┘                          │  |  │
                                ▼ MaxPool                                  │  │  │
                          ┌─────────────────────┐                          │  │  │
                          │ Residual Block ×2   │ conv4 (skip) ────---┐    │  │  │
                          │ (filters × 8)       │                     │    │  │  │
                          └──────────┬──────────┘                     │    │  │  │
                                ▼ MaxPool                             │    │  │  │
                          ┌─────────────────────--┐                   │    │  |  │
                          │ Residual Block ×2     │                   │    │  │  │
                          │ (filters × 16         │ ◀── BOTTLENECK    │    |  |  |
                          └──────────┬────────────┘                   │    │  │  │
                                ▼ UpSample                            │    │  │  │
                          ┌─────────────────────────┐                 │    │  │  │
                          │ Attention Gate ◀──────────────────────────┘    │  │  │
                          │ + Concatenate + Residual│                      │  │  │
                          │ Block (filters × 8)     │                      │  │  │
                          └──────────┬────────────--┘                      │  │  │
                                ▼ UpSample                                 │  │  │
                          ┌─────────────────────────┐                      │  │  │
                          │ Attention Gate ◀───────────────────────────────┘  │  │
                          │ + Concatenate + Residual│                         │  │
                          │ Block (filters × 4)     │                         │  │
                          └──────────┬──────────--──┘                         │  │ 
                                ▼ UpSample                                    │  │
                          ┌─────────────────────────┐                         │  | 
                          │ Attention Gate ◀──────────────────────────────────┘  │
                          │ + Concatenate + Residual│                            │
                          │ Block (filters × 2)     │                            │
                          └──────────┬──────────--──┘                            │
                                ▼ UpSample                                       │
                          ┌─────────────────────────┐                            │
                          │ Attention Gate ◀─────────────────────────────────────┘
                          │ + Concatenate + Residual│
                          │ Block (filters × 1)     │
                          └──────────┬──────────--──┘
                                     ▼
                          ┌─────────────────────┐
                          │ Conv2D(3, sigmoid)  │  ← predicted correction
                          └──────────┬──────────┘
                                     ▼
                          ┌────────────────────-─┐
                          │  Add(correction,     │  ← GLOBAL RESIDUAL LEARNING
                          │       original input)│     (model learns the delta,
                          └──────────┬──────────-┘      not the whole image)
                                     ▼
                          ┌─────────────────────┐
                          │  Clip to [0, 1]     │
                          └──────────┬──────────┘
                                     ▼
                              OUTPUT (384×384×3)
```

### Key architectural components

**Residual Block.** Each encoder/decoder stage uses a residual block: two `Conv2D → BatchNorm` layers, followed by CBAM attention (described below), with a skip connection adding the block's input back to its output before a final ReLU. This is the standard ResNet pattern, which helps gradients propagate cleanly through a network this deep without vanishing.

**Channel Attention (part of CBAM).** Inside every residual block, channel attention asks "*which feature channels matter most for this image?*" — it pools each channel down to a single number (via both global average pooling and global max pooling), passes that through a small shared MLP, and uses the result to re-weight every channel via a sigmoid gate. Channels that matter get amplified; channels that don't get suppressed.

**Spatial Attention (part of CBAM).** Immediately after channel attention, spatial attention asks the complementary question: "*which spatial locations matter most?*" — it pools across the channel dimension (average and max), concatenates those two maps, and runs a 7×7 convolution + sigmoid over them to produce a single spatial weighting mask applied back over the whole feature map. Together, channel and spatial attention let the network focus computation on both the right features and the right regions simultaneously, rather than treating every pixel and every channel as equally important.

**Attention Gate (on skip connections).** A plain U-Net copies encoder features straight across to the decoder via skip connections. This network instead gates each skip connection: the encoder's features and the decoder's upsampled features are both projected, summed, passed through a ReLU and a 1×1 convolution, and squashed through a sigmoid to produce an attention mask — the encoder features are then multiplied by that mask before being concatenated into the decoder. This lets the network suppress irrelevant or noisy encoder detail (a real risk in low-light images, where shadow regions carry mostly sensor noise) rather than blindly forwarding it.

**Global Residual Learning.** Rather than asking the final layer to output the entire enhanced image from scratch, the network outputs a *correction* (via a sigmoid-activated convolution) which is then added directly back onto the original input image, with the sum clipped to a valid `[0, 1]` range. This is a deliberate and meaningful design choice: it means the network only has to learn *what needs to change* — brighten this, recover color there — rather than reconstructing structure the input already mostly contains. This generally makes the optimization problem considerably easier and helps preserve fine detail that a from-scratch reconstruction might blur away.

## 5. Loss Function & Metrics

The model is trained with a **combined pixel-wise and structural loss**, rather than relying on a single metric:

```python
combined_loss = 0.4 * MSE(y_true, y_pred) + 0.6 * (1 − SSIM(y_true, y_pred))
```

- **MSE (Mean Squared Error)** penalizes raw pixel-value differences — good at driving overall brightness/color accuracy, but known to correlate poorly with how humans actually perceive image quality (it doesn't "know" anything about structure).
- **SSIM (Structural Similarity Index)** measures similarity in local luminance, contrast, and structure between two images — much closer to perceptual quality than raw pixel error, which is why it's weighted more heavily (0.6 vs. 0.4) in the combined loss.

Two additional metrics are tracked during training without contributing to the loss itself:

- **PSNR (Peak Signal-to-Noise Ratio)** — a standard image-restoration quality metric, reported in dB, where higher is better
- **SSIM**, also tracked directly as a metric (in addition to its role inside the loss), for an interpretable structural-similarity score during training

## 6. Training Configuration

| Setting | Value |
|---|---|
| Input resolution | 384 × 384 × 3 |
| Base filter count | 48 |
| Batch size | 4 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| LR schedule | Held constant for 5 epochs, then decayed ×0.8 through epoch 15, then ×0.8 every 8 epochs thereafter |
| Max epochs | 80 (with early stopping) |
| Early stopping | Monitored on validation PSNR, patience = 12 epochs, restores best weights |
| Checkpointing | Best model saved by validation PSNR (`best_low_light_model.keras`) |

## 7. Results & Evaluation

Training history (loss, PSNR, and SSIM over epochs, for both training and validation splits) is plotted and saved to `enhanced_training_history.png`. Qualitative results — side-by-side comparisons of the low-light input, the model's enhanced output (annotated with its PSNR against ground truth), and the actual ground truth — are saved to `result_comparison.png` via the `visualize_results()` function, sampling random examples from the validation set.

The final trained model is saved as a Keras model file (`denoisinsg_model.keras`) for downstream inference or deployment.

## 8. Project Structure

```
.
├── hackathon.ipynb              # Main notebook: data loading, model, training, evaluation
├── best_low_light_model.keras   # Best checkpoint by validation PSNR (saved during training)
├── denoisinsg_model.keras       # Final saved model
├── enhanced_training_history.png # Loss/PSNR/SSIM curves over training
└── result_comparison.png         # Qualitative low-light / output / ground-truth comparisons
```

## 9. How to Run

This notebook was developed and run on Kaggle, with GPU acceleration enabled and the LOL / LOL-v2 datasets attached as Kaggle dataset sources. To reproduce:

1. Attach the **LOL Dataset** (`soumikrakshit/lol-dataset`) and **LOL-v2 Dataset** (`tanhyml/lol-v2-dataset`) as data sources, or adjust the hardcoded paths (`lol_base`, `lolv2_base`) to point at local copies of these datasets.
2. Ensure a GPU runtime is enabled — training a U-Net at 384×384 resolution on CPU alone would be impractically slow.
3. Run all cells in order. Data loading and augmentation cells must execute before the model-definition and training cell.
4. The final cell saves the trained model to disk for reuse.

**Dependencies:** `tensorflow`, `numpy`, `opencv-python` (`cv2`), `matplotlib`, `imageio`, `scikit-learn`

## 10. Citations & References

- Chen Wei, Wenjing Wang, Wenhan Yang, Jiaying Liu. **"Deep Retinex Decomposition for Low-Light Enhancement."** *British Machine Vision Conference (BMVC), 2018.* — introduces the original LOL dataset used in this project.
- Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, Jiaying Liu. **"Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement."** *IEEE Transactions on Image Processing, 2021.* — introduces LOL-v2 (both Real_captured and Synthetic subsets).
- Olaf Ronneberger, Philipp Fischer, Thomas Brox. **"U-Net: Convolutional Networks for Biomedical Image Segmentation."** *MICCAI 2015.* — the encoder-decoder-with-skip-connections architecture this model's backbone is built on.
- Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. **"CBAM: Convolutional Block Attention Module."** *ECCV 2018.* — the channel + spatial attention mechanism used inside this project's residual blocks.
- Ozan Oktay et al. **"Attention U-Net: Learning Where to Look for the Pancreas."** *arXiv:1804.03999, 2018.* — the attention-gated skip connection design adapted for the decoder pathway here.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **"Deep Residual Learning for Image Recognition."** *CVPR 2016.* — the residual ("skip connection") block pattern used throughout the encoder and decoder.
- Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. **"Image Quality Assessment: From Error Visibility to Structural Similarity."** *IEEE Transactions on Image Processing, 2004.* — defines the SSIM metric used in both the loss function and evaluation.

---

*Note: this README was generated from the project's notebook contents. The dataset/architecture citations above are the original papers underlying the techniques actually used in the code (LOL/LOL-v2, U-Net, CBAM, attention gates, residual blocks, SSIM) — verify exact venue/year details against the original sources before submission if this is for formal academic credit, since they were reconstructed from general knowledge of these well-known papers rather than fetched live.*
Live Site: https://image-enhancer-app-eboppjjbgfavyvlxprzg8j.streamlit.app/
