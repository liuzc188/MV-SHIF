# MV-SHIF

This repository contains the code and data for the paper titled "**Multi-View Symmetric Hypothesis Inference Fusion
Network for Emotion-Cause Pair Extraction with Textual Entailment Paradigm**". The paper introduces a novel approach for
Emotion-Cause Pair Extraction using a Multi-View Symmetric Hypothesis Inference Fusion Network. This README provides an
overview of the repository and instructions for running the code and using the data.

## Contents

- [Requirements](#requirements)
- [Data](#data)
- [Usage](#usage)

## Requirements

To run the code in this repository, you'll need the following dependencies:

- Python 3.8
- PyTorch 1.8
- transformers

Install these dependencies using pip:

```shell
pip install -r requirements.txt
```

## Data

The data used for this project is available in the `ecpe_extractor/data/` directory. There are two data split
strategies, including 10 fold cross validation and train/valid/test.

## Usage

To train and test the model:

   ```
   bash training/run_experiment.sh
   ```
