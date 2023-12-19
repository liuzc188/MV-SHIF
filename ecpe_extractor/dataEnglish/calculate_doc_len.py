# @File ：calculate_doc_len.py

import json


def calculate_doc_len_stats(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    total_doc_len = 0
    max_doc_len = 0
    max_doc_id = None
    max_doc_pairs = None

    for document in data:
        doc_id = document["doc_id"]
        doc_len = document["doc_len"]
        pairs = document["pairs"]

        total_doc_len += doc_len

        if doc_len > max_doc_len:
            max_doc_len = doc_len
            max_doc_id = doc_id
            max_doc_pairs = pairs

    average_doc_len = total_doc_len / len(data)

    print("Average doc_len:", average_doc_len)
    print("Maximum doc_len:")
    print("   doc_id:", max_doc_id)
    print("   doc_len:", max_doc_len)
    print("   pairs:", max_doc_pairs)


def calculate_pair_to_clause_ratio(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    total_ratio = 0
    num_documents = len(data)
    print("num_documents:", num_documents)
    total_pair = 0
    total_doc_len = 0

    for document in data:
        pairs = document["pairs"]
        num_clauses = len(document["clauses"])
        expected_ratio = num_clauses ** 2  # 子句对数的平方

        actual_ratio = len(pairs) / expected_ratio
        total_ratio += actual_ratio

        total_pair += len(pairs)
        total_doc_len += expected_ratio

    average_ratio = total_ratio / num_documents
    total_ratio = total_pair/total_doc_len

    print("Average pair-to-clause ratio:", average_ratio)
    print("Total pair-to-clause ratio:", total_ratio)


json_file = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/data/all_data_pair.json'

calculate_doc_len_stats(json_file)
calculate_pair_to_clause_ratio(json_file)
