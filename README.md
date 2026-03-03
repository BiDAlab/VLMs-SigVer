# Kinematic hallucinations in vision-language models: A study on zero-shot signature verification

[cite_start]Official repository for the paper: **"Kinematic hallucinations in vision-language models: A study on zero-shot signature verification"**[cite: 2, 32].

[![Status](https://img.shields.io/badge/Status-Under_Revision-orange)](https://www.sciencedirect.com/journal/pattern-recognition-letters)

---

## 📌 About the Study

[cite_start]This repository contains the experimental framework for the first study evaluating the **zero-shot performance** of commercial Vision-Language Models (VLMs) —specifically **GPT-5.2** and **Gemini 2.5 Pro**— in the context of forensic signature verification[cite: 49, 77]. 



[cite_start]Our research identifies a significant **"Rationalization Trap"** [cite: 53, 123][cite_start]: while VLMs demonstrate exceptional geometric reasoning in random forgery scenarios, they are susceptible to **kinematic hallucinations** in skilled forgeries, where they fabricate motor-related dynamics (such as speed or pressure) to justify morphological decisions[cite: 53, 126, 275].

### Key Contributions:
* [cite_start]**Large-scale evaluation:** Analysis of **45,520 comparisons** across three diverse forensic tasks[cite: 38, 208, 247].
* [cite_start]**Kinematic-Encoded Representation:** A preprocessing framework that maps normalized pressure $P(t)$ to grayscale stroke intensity to assess its impact on model reasoning[cite: 50, 79, 150].
* [cite_start]**Kinematic Hallucination Index (KHI):** A new linguistic metric proposed to quantify the density of hallucinated kinematic descriptions in the models' forensic rationales[cite: 37, 80, 193].
* [cite_start]**Probabilistic Scoring:** Extraction of latent token probabilities (logprobs) to compute continuous similarity scores for EER analysis[cite: 51, 78, 177].

---

## 📝 Experimental Protocol

### Dataset & Tasks
[cite_start]We utilized the evaluation dataset of the **Signature Verification Challenge (SVC)**[cite: 142]:
* [cite_start]**Task 1 (Office - Stylus):** 6,000 comparisons with digital pen (pressure included)[cite: 143].
* [cite_start]**Task 2 (Mobile - Finger):** 9,520 comparisons via finger input[cite: 144].
* [cite_start]**Task 3 (Combined):** 12,000 comparisons for cross-device analysis[cite: 145].

### Signature Preprocessing
[cite_start]Since VLMs are blind to raw time-series data, we transform the signals into static images[cite: 147]. [cite_start]For Tasks 1 and 3, we implement a **pressure-encoded representation** where high-pressure segments are mapped to darker strokes, providing the VLM with visual grounding for movement analysis[cite: 150, 151].

### System Prompt
[cite_start]The models are instantiated as **Forensic Document Examiners**[cite: 159, 164]. [cite_start]We utilize a two-stage reasoning strategy to extract an initial visual impression followed by a reflective verdict after a Chain-of-Thought (CoT) phase[cite: 159, 168]:

```json
{
  "Role": "Forensic Document Examiner",
  "Task": "Analyze AI generated signatures, verifying if they belong to the same identity.",
  "Format": "Strict JSON",
  "Fields": [
    "Initial verdict: 'Same Identity' or 'Different Identity'",
    "Analysis: 'Technical comparison'",
    "Certainty: '0-100' (avoiding exactly 0 or 100)",
    "Final verdict: 'Same Identity' or 'Different Identity'"
  ]
}
