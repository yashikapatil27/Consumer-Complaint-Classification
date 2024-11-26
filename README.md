# Consumer-Complaint-Classification

This repository demonstrates a two-phase approach to text classification of consumer complaints. The project is designed to classify consumer complaints into predefined categories based on their content. 

## Project Structure

```plaintext
Consumer-Complaint-Classification/
│
├── CODE/
│   ├── phase1/
│   │   ├── 1_data_cleaning.py            # Raw data cleaning and preparation
│   │   ├── 2_eda.py                      # Exploratory Data Analysis (EDA)
│   │   ├── 3_no_preprocess_no_tuning.py  # Baseline model without preprocessing or tuning
│   │   ├── 4_pre-processing.py           # Text preprocessing and feature extraction
│   │   ├── 5_preprocess_no_tuning.py    # Preprocessing without hyperparameter tuning
│   │   ├── 6_hyperparameter_gridsearchcv.py # Hyperparameter tuning using GridSearchCV
│   │   ├── 7_test_prediction.py          # Testing and generating predictions
│   │   └── main_phase1.py                # Main file for Phase 1 (Training and evaluation)
│   │
│   └── phase2/
│       ├── 10_Feedforward.py             # Feedforward neural network model
│       ├── 11_Transformer.py             # Transformer model
│       ├── 8_StackedRNN.py              # Stacked RNN model
│       ├── 9_StackedLSTM.py             # Stacked LSTM model
│       └── main_phase2.py               # Main file for Phase 2 (Training and evaluation)
│
├── LOG FILE/                           # Folder for logging model training progress and results
│
└── README.md                           # Project documentation
```

**Phase 1 (Statistical Techniques):**
  - TF-IDF for feature extraction
  - Logistic Regression and Naive Bayes classifiers
  - Cross-validation for model evaluation
    
**Phase 2 (Deep Learning Architecture):**
  - Pretrained word embeddings (GloVe, Word2Vec) or custom embeddings
  - CNN, RNN, LSTM, and Transformer architectures for text classification
  - Hyperparameter tuning and model optimization.


