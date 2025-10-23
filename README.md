# MPLI-GNN: Multi-level Preference Local Interest-oriented Graph Neural Network for Session-based Recommendation

Implementation of the paper:

> **"Multi-level Preference Local Interest-oriented Graph Neural Network for Session-based Recommendation"**  
> (MPLI-GNN)

---

## Abstract
Session-based recommendation (SBR) is important in modern recommender systems due to its ability to model user preferences without requiring long-term user profiles. This capability is particularly valuable in scenarios involving anonymous users or rapidly changing user contexts, where long-term interaction histories are unavailable or unreliable. However, SBR faces several challenges. First, capturing users' diverse multi-layered interests through graph-based methods remains difficult. Second, interest shifts introduce noise within and across sessions. Third, effectively modeling users' immediate intent while preserving local interests presents challenges.

We propose the **Multi-level Preference Local Interest-oriented Graph Neural Network (MPLI-GNN)** to address these issues. Our approach incorporates three main components:
1. For **local interest representation**, we design an adaptive graph module with convolutional residual networks to capture fine-grained features.
2. For **global interest modeling**, we introduce a target node and combine graph convolution with sparse attention mechanisms to build a consistent representation of longer-term user tendencies.
3. Finally, our **intent fusion module** prioritizes local interests as the primary factor for recommendations.

Experimental evaluations on real-world datasets show that MPLI-GNN outperforms state-of-the-art methods. Ablation studies confirm the effectiveness of each proposed module. Our approach provides new insights and solutions for the development of the session-based recommendation field.

---

## Model Overview

<p align="center">
  <img src="assets/framework.png" width="600" alt="MPLI-GNN Framework">
</p>

**MPLI-GNN** integrates multi-level interest modeling and denoising mechanisms to enhance the accuracy and robustness of session-based recommendation.

### Core Components:
- **Adaptive Graph Module:** Captures fine-grained local interactions with residual graph convolution.  
- **Global Interest Module:** Models long-term stable preferences using sparse attention and a target node mechanism.  
- **Intent Fusion Module:** Combines global and local signals to generate final recommendations.

---

## Environment Setup

### Requirements
Python >= 3.8

PyTorch >= 1.10

NumPy >= 1.21

Pandas >= 1.3

scikit-learn >= 0.24



