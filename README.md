# More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.21298)

This repository introduces **H-VLI (Hate via Vision-Language Interplay)**, a benchmark dataset specifically curated to decipher "semantic intent shifts" in multimodal hate speech, where toxicity often emerges from the subtle interplay between benign modalities. Alongside the dataset, we provide the official implementation of **ARCADE (Asymmetric Reasoning via Courtroom Agent DEbate)**, a hierarchical courtroom framework designed to scrutinize these complex cross-modal interactions and uncover latent hateful intents.

---

## 🚩 News
- **[2026-04-06]**: Our paper has been accepted by **ACL 2026 Findings**! 🎉

---

## 1. Dataset: H-VLI Benchmark

### 1.1 Data Preparation

#### Download Images
To run the experiments, please download the images for the respective datasets and place them in the following directory structure:

1.  **FHM Images**: Download from [Kaggle](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset).
2.  **MMHS150K Images**: Download from the [official site](https://gombru.github.io/2019/10/09/MMHS/).
3.  **H-VLI Images**: Download from [Google Drive](https://drive.google.com/file/d/1HdAck-PB9PW8BTTHhizylPODVqCxtoEF/view?usp=drive_link).

#### Dataset Splits
The standard train/test splits for the H-VLI benchmark are provided in the `data/` directory (`train_set.json` and `test_set.json`).

#### Directory Structure
Organize the downloaded images as follows:
```text
imgs/
├── FHM/                 # .jpg files from Facebook Hateful Meme
├── MMHS150K/            # .jpg files from MMHS150K
└── H-VLI_images/        # .jpg files from H-VLI
```

### 1.2 Dataset Introduction

Motivated by the evolution of hate speech from plain text to complex multimodal expressions, traditional binary detection often fails to identify implicit attacks. To address this, the **H-VLI (Hate via Vision-Language Interplay)** mechanism focuses on capturing "semantic intent shifts." Within this framework, uni-modal annotations are interpreted in isolation, where an individual text or image might appear entirely benign or overtly toxic. However, through intricate inter-modality interaction, these modalities combine to create a semantic shift—either constructing implicit hate from benign unimodal cues or neutralizing apparent toxicity through semantic inversion. 

<p align="center">
  <img src="assets/sample_1.png" width="45%" />
  <img src="assets/sample_2.png" width="45%" />
  <br>
  <em>Figure 1: Typical examples of "Implicit Hate" in H-VLI, where toxicity emerges solely from the interplay between benign-looking text and images.</em>
</p>

To strictly correspond with the H-VLI mechanism, the benchmark dataset is constructed using a hybrid pipeline of consensus filtering, generative injection, and human-in-the-loop annotation. This guarantees a high density of challenging multimodal samples where the true intent fundamentally hinges on the intricate interplay of modalities rather than relying on isolated visual or textual slurs.

We introduce the **H-VLI (Hate via Vision-Language Interplay)** benchmark, specifically curated to challenge models with subtle cross-modal interactions.

<p align="center">
  <img src="assets/dataset_construction.jpg" width="80%" />
  <br>
  <em>Figure 2: The construction pipeline of the H-VLI dataset, combining real-world sampling with generative injection.</em>
</p>

#### Annotation and Difficulty Stratification
To capture the complexity of multimodal hate, particularly when modalities conflict, we introduce the **Stratified Multimodal Interaction (SMI)** paradigm. For each sample, we annotate a **five-tuple**, explicitly labeling unimodal sentiments alongside the final multimodal annotation: 

$$ \mathcal{A}_i = (y_i^{\text{text}}, e_i^{\text{text}}, y_i^{\text{image}}, e_i^{\text{image}}, y_i^{\text{combined}}) $$

where $y_i^{\text{text/image}}$, $e_i^{\text{text/image}}$ denote the unimodal labels and explanations respectively. $y_i^{\text{combined}}$ represents the final multimodal ground-truth label.

**Taxonomy of Multimodal Interaction:**
Under the SMI paradigm, the interplay between unimodal signals ($y_{\text{text}}, y_{\text{image}}$) and the combined outcome ($y_{\text{combined}}$) yields eight distinct patterns ($(y_{\text{text}}, y_{\text{image}}, y_{\text{combined}}) \in \{0,1\}^3$, where 1 denotes toxicity and 0 denotes benign). Based on reasoning complexity, we categorize them into three levels:

- **(1) Low Complexity (Aligned/Dominant):** Covers explicit cases requiring minimal cross-modal deduction. This includes purely benign (0,0,0), redundant hate (1,1,1), and unimodal dominance (1,0,1 or 0,1,1).
- **(2) Medium Complexity (Contextual Neutralization):** Includes patterns where toxicity in one modality is mitigated by the benign context of the other (1,0,0 or 0,1,0), requiring the model to recognize how context neutralizes apparent slurs.
- **(3) High Complexity (Emergent Semantic Shift):** Strictly tests the H-VLI benchmark via synergistic hate (0,0,1) and dual-inversion (1,1,0). These demand deep inferential reasoning to resolve cases where the final label contradicts both unimodal signals—either detecting implicit attacks emerging from benign cues (0,0,1) or recognizing how apparent toxicity is neutralized through complex cross-modal irony or counter-speech (1,1,0).

<p align="center">
  <img src="assets/showcases.jpg" width="60%" />
  <br>
  <em>Figure 3: Showcase of different interaction patterns in H-VLI.</em>
</p>

<p align="center">
  <img src="assets/dataset_distribution.jpg" width="60%" />
  <br>
  <em>Figure 4: Statistical breakdown of the H-VLI dataset.</em>
</p>

---

## 2. Methodology: ARCADE Framework

**ARCADE (Asymmetric Reasoning via Courtroom Agent DEbate)** simulates a judicial process to decipher multimodal intent shifts.

### 2.1 Framework Overview
<p align="center">
  <img src="assets/ARCADE.jpg" width="80%" />
  <br>
  <em>Figure 5: The architecture of the ARCADE framework, featuring a Gated Dual-Track mechanism for explicit and implicit hate detection.</em>
</p>

ARCADE employs a **Gated Dual-Track Mechanism** to efficiently process multimodal samples:

1.  **Rapid Scan**: Every sample first undergoes a preliminary screening by the **Prosecutor** agent.
2.  **Track I: Fast-Track Trial (Explicit Hate)**: If the initial scan detects overt hateful cues, the sample is routed to Track I. It undergoes a single-round adversarial exchange before the **Judge** renders a verdict.
3.  **Track II: Deep-Dive Trial (Implicit Hate)**: If no explicit hate is found but latent risks are suspected, the sample enters Track II. It undergoes **$K$ rounds** of intensive debate between the Prosecutor and Defender to uncover subtle intent shifts before final adjudication.
4.  **Summary Dismissal**: If the Prosecutor finds no evidence of hate in its assessment, a Summary Dismissal can be triggered, ruling the sample as non-hateful without further debate.

**Core Roles:**
- **Prosecutor (Risk Discovery)**: Operates under a "presumption of guilt," actively hypothesizing malice and uncovering latent hate in metaphors and symbols.
- **Defender (Contextual Safety)**: Operates under a "presumption of innocence," scrutinizing evidence for benign motivations like satire or counter-speech.
- **Judge (Final Arbiter)**: Evaluates the adversarial exchange to render a final verdict and provide a natural language explanation.

### 2.2 Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Rename `.env.example` to `.env` and fill in your API keys.

#### Key Management Rules:
1. **Priority**: GPT and Gemini models will prioritize official APIs if `OPENAI_API_KEY` or `GEMINI_API_KEY` is provided.
2. **Auto-Fallback**: If official keys are missing, the system automatically attempts to use alternative providers (e.g., `API_YI_API_KEY`).
3. **Key Polling**: For DashScope (Qwen), GLM, and API_YI, you can configure multiple keys (e.g., `KEY_1, KEY_2`) to balance rate limits.

### 3.3 Experimental Guide

#### Configuration
Before running the experiments, you can customize the execution by modifying the following parameters in `main.py`:

```python
# Select the models for inference
MODELS_TO_RUN = ['qwen3-vl-plus']  # Model(s) acting as the Judge
AUX_MODEL = "qwen3-vl-plus"        # Model acting as the Prosecutor and Defender

# Specify the input dataset path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_PATH = os.path.join(BASE_PATH, "data/test_set.json")
```

#### Basic Commands

```powershell
# 1. Run ARCADE hierarchical debate system (Default)
python main.py --run_mode ARCADE --samples 100

# 2. Run direct classification baseline (Baseline None)
python main.py --run_mode none --samples 100 --class_mode binary
```

#### Argument Descriptions

| Argument | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run_mode` | `ARCADE`, `none` | `ARCADE` | **Experiment Mode**. `ARCADE`: Hierarchical debate; `none`: Direct inference. |
| `--class_mode` | `multiclass`, `binary` | `multiclass` | **Classification Standard**. `multiclass`: 0-5 labels; `binary`: 0-1 labels. |
| `--samples` (`-s`) | Integer | `10` | Number of samples to test. Set to 0 for the full dataset. |
| `--threads` | Integer | `16` | Number of concurrent threads for API requests. |
| `--rounds` | Integer | `3` | Number of debate rounds for the implicit detection track. |
| `--seed` | Integer | `2024` | Random seed for data sampling. |

### 2.4 File Structure
- `data/`: Directory containing dataset splits (`train_set.json`, `test_set.json`) and all sample metadata including tweet text and labels.
- `imgs/`: Directory containing source images for FHM, MMHS150K, and H-VLI.
- `main.py`: Main entry point for data sampling, concurrent scheduling, and evaluation.
- `court_system.py`: Core system logic implementing the ARCADE hierarchical routing.
- `court_prompts.py`: Agent prompt templates for the multi-class categorization task.
- `court_prompts_binary.py`: Agent prompt templates for the binary detection task.
- `llm_client.py`: API client supporting official direct connections and provider-based fallbacks.
- `evaluator.py`: Logic for calculating Accuracy, Macro-F1, and other performance metrics.
- `utils.py`: Utility functions for data loading, sampling, image encoding, and file operations.

---

## 3. Results Output
- Results are stored in `answers_system/{class_mode}/{run_mode}/{timestamp}/{model}/`.
- `results_{model}.json`: Detailed inference logs for every sample.
- `report.txt`: Summary report including global metrics and difficulty-wise performance.

---

## License
The H-VLI dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Users must adhere to the terms of source datasets (MMHS150K, FHM).

## Citation
If you find our work helpful, please cite us:

```bibtex
@article{sun2026sumpartsdecipheringintent,
  title={More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection},
  author={Runze Sun and Yu Zheng and Zexuan Xiong and Zhongjin Qu and Lei Chen and Jiwen Lu and Jie Zhou},
  journal={arXiv preprint arXiv:2603.21298},
  year={2026} 
}
```

---

## Disclaimer
This repository contains examples of hateful or offensive content. These materials are provided strictly for academic research purposes, specifically for the development and evaluation of multimodal hate speech detection systems. The authors of this work do not condone, support, or agree with any hateful sentiments, stereotypes, or offensive views expressed in these samples.