# Music Genre Classification

**Authors:** Jonathan Neimann, Matt Maslow  
**Date:** December 6, 2024  
**Course:** DS 452 – Applied Machine Learning  

---

## Project Overview

This project implements a music genre classification system using two modeling approaches:

- A traditional **k-Nearest Neighbors (KNN)** classifier trained on handcrafted audio features.
- A **Convolutional Neural Network (CNN)** trained on mel-spectrogram image representations.

By comparing these models on the GTZAN dataset, we demonstrate the strengths and limitations of feature-based vs. deep learning-based methods for music information retrieval.

---

## Motivation

Music genre classification is a fundamental task in music recommendation, playlist generation, and media organization. However, genre boundaries can be ambiguous, and audio data is complex and high-dimensional. This project seeks to explore how well traditional versus deep learning approaches can tackle this problem.

---

## Dataset

**GTZAN Genre Collection**  
- 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock  
- Audio format: `.wav` files at 22,050 Hz  
- Two types of input used:
  - **30-second clips**: used for both feature extraction and spectrogram generation.
  - **3-second segments**: used primarily to augment CNN training.

---

## Methodology

### 1. Feature Extraction for KNN
- Extracted from 30-second audio clips using `librosa`
- Features used:
  - **MFCCs** (Mel Frequency Cepstral Coefficients)
  - **Chroma Features**
  - **Spectral Contrast**
- Averaged over time to form a fixed-length vector per song.

### 2. Spectrograms for CNN
- Mel-spectrograms generated and log-transformed
- Resized to **128x128** pixels
- Used as image input to the CNN

### 3. CNN Architecture
- Multiple convolution and max-pooling layers
- Dense output layer with softmax activation
- Loss: Categorical Cross-Entropy  
- Optimizer: Adam  
- Trained for 10 epochs on spectrogram images

---

## Results

### Convolutional Neural Network (CNN)
- **Training Accuracy:** 98.52%  
- **Validation Accuracy:** 98.99%  
- **Validation Loss:** 0.0434  
- Strong generalization across all genres  

### k-Nearest Neighbors (KNN)
- **Accuracy:** 90%  
- **Macro Avg F1-Score:** 0.90  
- Strong performance on distinct genres like Blues and Country  
- Struggled with overlapping genres like Jazz and Hip-Hop  

---

## Analysis

- The **CNN outperformed** the KNN in every metric, especially in its ability to generalize and capture subtle audio features via spectrograms.
- The **KNN** model showed strong performance with well-separated genres but was limited by the handcrafted nature of features.

---

## Conclusion

The deep learning approach (CNN) significantly outperforms the traditional KNN classifier for music genre classification. This highlights the power of spectrogram-based image input and automatic feature learning in complex audio tasks. Future work could explore larger datasets, ensemble models, and alternative CNN architectures for further performance gains.

---

## References

1. Choi, K., et al. (2017). “Convolutional Recurrent Neural Networks for Music Classification.” *ICASSP 2017*. [IEEE DOI](https://doi.org/10.1109/ICASSP.2017.7952585)  
2. Tzanetakis, G., & Cook, P. (2002). “Musical Genre Classification of Audio Signals.” *IEEE Transactions on Speech and Audio Processing*. [PDF](http://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)
