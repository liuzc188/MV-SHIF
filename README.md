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

## Other

1. 使用其他预训练模型时，一定要修改preprocess_tokenized_data_v1.py等文件中的BertTokenizer，可以统一改为AutoTokenizer。同理，BertModel也应改成AutoModel；
2. 应注意process_data_v1.py文件中的截断函数，中文ECPE数据集可使用375，英文时，会报错。此处使用longformer作为预训练模型；
3. 直接使用longformer，在英文数据集上会出现OOM问题。此处简单地修改了longformer的config.json文件中的如下参数：
    ```
     "attention_window": [
      512
      ],
     "num_attention_heads": 1,
     "num_hidden_layers": 1,
   ```
4. batch_size = 12时，可占满3090的显存；
5. 英文json数据已经过筛选，剔除异常样本（>370），不需要再改动；
