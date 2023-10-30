# Text_Mining - NLTK

## Introduction

This repository contains an individual project report that explores a dataset of conversations from the British National Corpus. The project focuses on data mining principles and the application of deep learning techniques for text mining. Additionally, it investigates, models, and reports insights from the corpus data.

## Jupyter Notebook

You can access the Jupyter Notebook containing the code and documentation for this project by clicking[GEM_T5.ipynb](https://github.com/NikhilaRamisetti/Text_Mining-NLTK/blob/main/GEM_T5.ipynb).

## Task 1: Description of Data and Methods

Task 1 involves exploring the British National Corpus (BNC) 2014 spoken corpus dataset, pre-processing the data, and extracting insights. 

Pre-processing techniques, such as tokenization, stemming/lemmatization, and handling missing values, are applied to the dataset.

## Task 2: Investigation, Modeling, and Insights

Task 2 focuses on investigating the dialogue data, summarizing topics, and creating models. We analyze the following:
**Feature Extraction and Analysis:**
- Age demographics and average utterance length per speaker are extracted.
- Clustering algorithms (K-means, DBSCAN) group similar utterances.
- Named entity identification and sentiment analysis provide additional insights.

**Deep Learning Approaches:**
- Discusses common deep learning techniques: text classification, sentiment analysis, language modeling, and summarization.
- Highlights popular models (T5, BERT, GPT) and emphasizes the need for large datasets for training.

**Machine Learning Evaluation:**
- Explains the evaluation of machine learning systems for text data, covering metrics like accuracy, precision, recall, F1 score, and human evaluation for fluency and fidelity.

**Data Visualization:**
- Presents visualizations, including word frequency analysis across age groups.
- Analyzes vocabulary overlap frequency between speakers.
- Describes summarization techniques (extractive and abstractive).

**Machine Learning Models:**
- Trains logistic regression and random forest classifiers to predict age based on average utterance length.
- Utilizes logistic regression for vocabulary overlap prediction based on age.



Please refer to the project documentation and reports in the folder `docs/`


## Bonus Optional Tasks

Data pre-processing techniques for untagged XML files are detailed. This includes tokenization, part-of-speech chunking, named entity recognition, and collocation finding.


## Usage

To run this project, ensure you have the required libraries and tools installed, including NLTK, scikit-learn, and Transformers.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Text_Mining-NLTK.git
   ```
2. Dependencies

Before running the Jupyter Notebook, make sure to install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the project by executing the main script
 `src/`: The source code is the Jupyter Notebook containing the code and documentation for this project by clicking [GEM_T5.ipynb](https://github.com/NikhilaRamisetti/Text_Mining-NLTK/blob/main/GEM_T5.ipynb).

4. Refer to the results and detailed project documentation below:
- `results/`: Stores the project results and visualizations.

## Acknowledgments

- The project relies on various open-source libraries, including NLTK, scikit-learn, and Transformers.
- The dataset used in this project is from the British National Corpus.
