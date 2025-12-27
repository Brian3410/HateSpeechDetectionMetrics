# Hate Speech Detection Models - Performance Metrics

This repository contains comprehensive performance metrics and evaluation results for multiple hate speech detection models trained across various datasets and enhancement techniques.

## Repository Contents

```
Model Metrics.xlsx    # Complete experimental evaluation results
README.md            # This file
```

## Overview

The `Model Metrics.xlsx` file presents comprehensive performance metrics for all model architectures across four datasets under different enhancement technique configurations. Values outside parentheses represent baseline performance, while values in parentheses show performance after applying the respective enhancement technique.

## Datasets Evaluated

1. **Hate Corpus** - Multi-platform hate speech dataset
2. **Gab & Reddit** - Social media hate speech from Gab and Reddit platforms
3. **Stormfront** - White supremacist forum discussions
4. **Merged Dataset** - Combined dataset from all three sources

## Models Evaluated

- **Delta TF-IDF** - SVM-based classifier using delta TF-IDF features with 10-fold cross-validation
- **DistilBERT** - Distilled BERT model fine-tuned for hate speech detection
- **RoBERTa** - Robustly optimized BERT approach with domain-specific fine-tuning
- **Gemma-7B** - Google's 7 billion parameter model with LoRA fine-tuning
- **DeBERTa** - Decoding-enhanced BERT with disentangled attention mechanism
- **GPT-OSS 20B** - Open-source 20 billion parameter GPT model

## Enhancement Techniques

Each model was evaluated with the following enhancement configurations:

### Baseline Configuration
**Standard Training** - Establishes performance benchmarks using each model's default architecture without additional enhancement techniques. Employs standard training procedures over 10 epochs with stratified data splits (80% training, 10% validation, 10% testing) using random seed 42 for reproducibility. For Delta TF-IDF, uses 10-fold stratified cross-validation on the training data to simulate the 10-epoch training of neural models.

### Class Imbalance Handling
**SMOTE & Weighted Loss** - Synthetic Minority Over-sampling Technique (SMOTE) applied to training data combined with class-weighted loss functions. SMOTE resampling uses intelligent TF-IDF vectorization (1000 features, unigrams+bigrams) with cosine similarity-based synthetic sample mapping. Dynamically computed class weights adjust loss function penalties to account for class frequency disparities. Neural architectures use WeightedTrainer for weighted cross-entropy loss, while traditional models integrate balanced sampling strategies.

### Linguistic Enhancement
**POS Feature Integration** - Incorporates grammatical structure awareness through systematic extraction of linguistic ratio features including noun, verb, adjective, adverb, pronoun, and interjection ratios, plus profanity and capitalization patterns using spaCy's English language model. For neural architectures, POS features undergo dimensionality projection and concatenation with model representations through additional classifier layers. Traditional models integrate these features directly into their feature vectors. Addresses position bias issues in token classification by leveraging grammatical patterns indicative of hate speech discourse.

### Comprehensive Balancing
**SMOTE & Weighted Loss & POS Integration** - Combined application of SMOTE oversampling, weighted loss optimization, and POS feature integration. Addresses both class imbalance and linguistic feature representation simultaneously, providing comprehensive enhancement through integrated synthetic oversampling, weighted loss penalties, and grammatical structure awareness.

### Data Diversity Enhancement
**Text Data Augmentation (TDA)** - Implements sophisticated multi-layered text transformation techniques exclusively during training to enhance dataset diversity and model robustness. The augmentation framework combines:
- **Word-level transformations**: WordNet synonym replacement, random swapping/insertion/deletion with hate-speech-aware preservation
- **Character-level modifications**: Keyboard neighbor replacement, spelling error simulation, realistic typo introduction
- **Sentence-level restructuring**: Contraction transformation, structural variations
- **Contextual augmentation**: Transformer-based paraphrasing with semantic similarity validation

Quality control mechanisms preserve important hate speech terms and group identifiers while generating additional minority class examples through text variations to balance the dataset.

## Evaluation Metrics

Each model-dataset-technique combination is evaluated using the following metrics:

### Primary Metrics
- **Accuracy** - Overall classification accuracy (correct predictions / total predictions)
- **Macro F1** - Unweighted average F1 score across all classes (treats all classes equally)
- **AUC** - Area Under the ROC Curve (model's ability to distinguish between classes)

### F-Beta Scores
- **F0.5 Score** - Emphasizes precision over recall (β=0.5) - useful when false positives are costly
- **F2 Score** - Emphasizes recall over precision (β=2) - useful when false negatives are costly

### Weighted Metrics
- **Weighted F1** - F1 score weighted by class support (accounts for class imbalance)
- **Weighted Precision** - Precision weighted by class support
- **Weighted Recall** - Recall weighted by class support

## Results Format

Results are presented in the following format:

**Base Score (Enhanced Score)**

- **Base Score**: Performance of the model without the enhancement technique
- **Enhanced Score** (in parentheses): Performance after applying the enhancement technique

### Example Interpretation
```
Accuracy: 0.85 (0.89)
```
This indicates:
- Baseline accuracy: 85%
- Accuracy after enhancement: 89%
- Improvement: +4 percentage points

## Training Environment

All models were trained using **Monash M3 High Performance Computing** cluster:

- **GPU**: NVIDIA A40
- **Memory**: 128GB RAM
- **CPUs**: 8 cores
- **Training Time**: 3-24 hours per model (varies by architecture)
- **Framework**: PyTorch with HuggingFace Transformers