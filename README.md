# FAERS Adverse Event Signal Detection using Hybrid Semantic PRR

# 🔎 Project Overview

This project detects potential safety signals between drugs and adverse events using real-world pharmacovigilance data from **FAERS** (FDA Adverse Event Reporting System).

It enhances traditional **PRR (Proportional Reporting Ratio)** signal detection by combining:

*   📊 **Statistical disproportionality analysis**
*   🧠 **Semantic similarity** (BERT-based medical embeddings)
*   🔗 **Co-occurrence similarity** (drug-event behavior patterns)
*   ⚡ **FAISS-based nearest neighbor search** for scalability

The result is a **Hybrid Semantic Pooled PRR model** that improves signal detection accuracy while remaining memory-efficient.

---

# 🧠 What Problem Does This Solve?

Traditional PRR only checks **exact drug-event pairs**.

**Example:**
If "Heart Attack" is reported frequently with a drug, PRR detects it.

But what if reports use:
*   "Myocardial Infarction"
*   "Cardiac Event"
*   "Acute Coronary Syndrome"

Traditional PRR treats them separately.

This model intelligently groups related medical events using:
1.  **Semantic similarity** (meaning-based similarity)
2.  **Co-occurrence similarity** (how similarly they behave across drugs)

Then it **pools evidence** across related events. This improves detection of real safety risks.

---

# 📂 Project Structure

```text
FAERS_AE_MODEL/
│
├── semantic_prr.py              # Core Hybrid PRR model
├── validation_cases.py          # Known drug-event validation set
├── semantic_analysis.ipynb      # Embedding & similarity experiments
├── validation_analysis.ipynb    # PRR evaluation notebook
├── bert.ipynb                   # Embedding generation
└── README.md

```
#  ⚙️ Methodology
 ###   **1️⃣ Baseline PRR**
 PRR (Proportional Reporting Ratio)  measures how disproportionately a drug is associated with an event.
* ```text PRR ≥ 2 ``` is typically considered a safety signal.
 ###  **2️⃣ Semantic Similarity (Meaning Similarity)**
Uses SapBERT (medical language model) to convert event names into numerical vectors (embeddings).
* Example: "Heart Attack" and "Myocardial Infarction" → high cosine similarity.
 ###  **3️⃣ Co-Occurrence Similarity**
Events are represented as vectors of how often they appear with each drug.
Events that behave similarly across drugs are considered related.
FAISS is used for efficient nearest neighbor search.
### 4️⃣ Hybrid Similarity
The final similarity score balances meaning similarity and real-world reporting behavior:
math
```text 1 Hybrid Similarity = α × Semantic Similarity + (1 − α) × Co-occurrence Similarity ```
### 5️⃣ Pooled PRR
Instead of using only exact event counts, we pool evidence from similar events:
math
```text 1 pooled_cases = base_cases + Σ(similarity × neighbor_cases)```

Then, PRR is computed using these pooled counts.

# 📊 Validation

The model is validated on known drug-event safety pairs, including:

| Drug       | Adverse Event       |
|------------|----------------------|
| Metformin  | Lactic Acidosis     |
| Clozapine  | Agranulocytosis     |
| Warfarin   | Haemorrhage         |
| Linezolid  | Serotonin Syndrome  |

Validation logic is implemented in `validation_cases.py`.

## 🚀 How to Run

### 1️⃣ Install Dependencies
### 2️⃣ Prepare Data
  Ensure the following files are present in the results_private/ directory:
    * drug_event_pairs.parquet
    * ae_embeddings_sapbert.npy
    * unique_aes.txt
### 3️⃣ Run Model
```bash python semantic_prr.py```

# 📈 Example Output

The model produces:
* PRR
* Pooled cases
* Chi-square statistic
* Neighbor events used
* Signal decision (True/False)

# 🧪 Performance Improvements

### Optimizations included:
* Event frequency filtering (MIN_EVENT_FREQ)
* Float32 memory optimization
* FAISS nearest neighbor search
* Removal of full similarity matrix
* Sparse-like behavior using KNN graph
This prevents OOM errors and scales to large datasets.

# 🎯 Key Contributions

* ✔ Hybrid statistical + semantic model
* ✔ Memory-optimized architecture
* ✔ FAISS acceleration
* ✔ Validation against known FDA warnings
* ✔ Scalable design

📌 Future Improvements

* Dynamic alpha tuning
* Time-aware signal detection
* Graph-based clustering
* Confidence interval estimation
* Bootstrap validation
* SHAP-style explainability


