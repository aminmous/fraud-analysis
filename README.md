# fraud-analysis

## Description
This repository contains the Script dedicated to the analysis of the Dataset created using the "fraud webscraper" Repository. And can be used to replicate the experiments in the Masers Thesis "Machine Learning Methods for Text Based Fraud Detection".

## Content

- fraud.csv is the dataset that contains all variables for full list see Report repository
- amalgamate.py is the cript that combines the raw components from the webscraper repository into the dataset that can be utlized for the experiments
- classifier cotnains the script to run xgboost and random forest on whatever input necessary and a supplementary script
    * classifier.py is the script with the classifers
    * tabularize.py turns the relevant tabular features into the right format to run in classifier.py
- LDA is the file that cotnains the experiments revolving around topic extraction using LDA on the MDAs
    * lda_tuner.py is the script that contains the tuning of the LDA meaning the evaluation of which topic number is the most fitting for different subsets
    * lda_classifier.py is the script in which the lda model is tested based on the configuration from lda_tuner.py
- Modern-BERT contains all experiments and supplementary scripts revolving around "answerdotai/ModernBERT-base" on huggingface
    * MBERT_embeddings.py is the script with which the embeddings on the MDAs were extracted
    * MBERT_expansion.py is the script that contains the Modern-BERT tuning
    * expanding_sets.py contains the function with which the tuning script is run with different expanding sets extracted from the fraud.csv dataset
- PU_Learning contains the script in which the the PU learning is executed