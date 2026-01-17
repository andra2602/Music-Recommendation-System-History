# 🎵 Reconstructing the History of Music Recommendation

### *Content-Based • Collaborative Filtering • Neural Item2Vec*

This project reconstructs the historical evolution of music recommendation systems by implementing three major paradigms: **Content-Based Filtering**, **Collaborative Filtering (SVD)**, and **Neural Item2Vec embeddings**.
Using the *Kaggle 30K Spotify Songs* dataset, we reproduce the core ideas that shaped modern recommendation engines and compare how each paradigm interprets musical similarity.

## Project Overview

Music recommendation systems evolved from **hand-crafted feature similarity**, to **statistical models based on collective behavior**, and finally to **neural contextual embeddings**.
Our goal was to rebuild these three eras from scratch, under a unified experimental framework.


### Dataset & Preprocessing

We use the **Kaggle 30,000 Spotify Songs** dataset (≈30k tracks after cleaning).
Each track contains 10 normalized audio features (danceability, energy, valence, tempo, etc.) along with metadata such as artist and playlist genre.

Key preprocessing steps:

* removing duplicates & missing values
* normalizing audio features (Min–Max)
* preparing interaction matrices & synthetic playlists

(see dataset overview in the *slides, page 5* )

# Models Implemented

### 1. Content-Based Filtering

Uses **cosine similarity** over normalized audio features to recommend songs with similar sound profiles.

**Highlights**

* requires no user history
* fast and fully interpretable
* PCA used for feature-space visualization

PCA clusters shown in *slides, page 6* .

### 2. Collaborative Filtering (SVD)

We simulate user–song interactions to build a user–item matrix, then apply **Matrix Factorization** (SVD) to learn latent taste factors.

**Highlights**

* predicts missing preferences
* captures behavioral similarity (co-listening patterns)
* stronger personalization than CBF

Example recommendation outputs shown on *slides, page 7* .

### 3. Neural Recommendation — Item2Vec

A Word2Vec-style model trained on playlists, treating each playlist as a “sentence” and each track as a “word.”

We trained two variants:

1. **Mixed playlists** – captures broad compatibility
2. **Genre-structured playlists** – yields clearer embedding clusters

t-SNE visualization of learned embeddings is shown on *slides, page 8* .

# Results & Insights

Across the three paradigms, similarity is defined differently:

| Paradigm                    | Similarity Basis | Strengths                          | Limitations                 |
| --------------------------- | ---------------- | ---------------------------------- | --------------------------- |
| **Content-Based**           | Audio features   | Interpretable, fast                | No personalization          |
| **Collaborative Filtering** | User behavior    | Personalized, learns hidden tastes | Needs real interaction data |
| **Item2Vec (Neural)**       | Playlist context | Captures vibe / mood               | Needs large real playlists  |

**Key insight:**

> Combining multiple paradigms yields a more robust recommender — each captures a unique facet of musical similarity.
> (See *Conclusion section in report* )

# Limitations

As described in both the PPT and report:

* no real user logs → CF trained on simulated data
* playlists had to be **synthetically generated**
* audio features limited to 7 descriptors
* simplified models compared to industrial recommender systems
* evaluation restricted to internal consistency (PCA, cos-sim, t-SNE)

(see *Limitations, slides page 9*  and *report section 3* )


# Design

The project features an interactive user interface built with **Gradio**, created to transform the implemented recommendation algorithms into an intuitive, exploratory application.

The interface is structured to mirror the **historical evolution of recommender systems**, with each paradigm exposed through a dedicated tab:

- **Stage 1 – Content-Based Filtering**  
  Focuses on interpretable, feature-based similarity using audio descriptors and cosine distance.

- **Stage 2 – Collaborative Filtering**  
  Organized into multiple sub-tabs to reflect different behavioral assumptions, including:
  SVD-based latent factor recommendations, user–user similarity, and co-listening patterns.

- **Stage 3 – Neural / Contextual Filtering**  
  Enables exploration of semantic relationships through Word2Vec embeddings, including
  song similarity, genre compatibility, and semantic transition playlists.

The UI design emphasizes a **clear separation between model computation and user interaction**.
Each component follows a step-by-step interaction pattern (explicit inputs and structured outputs),
supporting **exploration, explanation, and direct comparison** between recommendation paradigms.

*(see slides pages 10–12 and report section 4)*


# Future Work

From both team members’ reflections and project goals:

* train on **realistic large-scale playlists** (e.g., Million Playlist Dataset)
* implement **hybrid recommendation systems**
* extend analysis to **multimodal signals** (lyrics, emotional embeddings)

(*report, section 5* )

# Contributions

### **Andra Mihaela Andruță**

* dataset preprocessing & cleaning
* exploratory data analysis
* CBF implementation
* PCA & word cloud analysis
* Word2Vec training + semantic playlist transitions
* co-design and implementation of the interactive Gradio user interface

(*slides page 3* )

### **Ioana Alexandra Tunaru**

* playlist corpus generation
* CF (SVD) implementation
* top-N recommendations
* popularity vs personalization evaluation
* comparative study CBF vs CF
* co-design and implementation of the interactive Gradio user interface

(*slides page 4* )


# How to Run

All experiments were executed in **Google Colab**.

### Install dependencies:

```bash
pip install numpy pandas scikit-learn gensim scikit-surprise matplotlib seaborn
```

### Run notebook:
`KaggleProiectAMI.ipynb`

No GPU is required — all models train in seconds.


# Contact

For academic use or questions:

* **Andra Mihaela Andruță** – [andra-mihaela.andruta@s.unibuc.ro](mailto:andra-mihaela.andruta@s.unibuc.ro)
* **Ioana Alexandra Tunaru** – [ioana-alexandra.tunaru@s.unibuc.ro](mailto:ioana-alexandra.tunaru@s.unibuc.ro)
