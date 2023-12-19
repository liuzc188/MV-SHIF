import json
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
sys.path.append('/home/ubuntu/workspace1/rmj/ECPE')
from ecpe_extractor.artifacts.config import *


def truncate_document(doc_len, doc_couples, clauses, emotion_categorys, emotion_tokens):
    """This function is used to handle situations where the length of a document exceeds a certain threshold."""

    # Calculate the total length of clauses in the document
    total_length = 0
    for clause in clauses:
        total_length += (2 + len(clause))

    # If the total length exceeds a predefined threshold, truncate
    # if total_length > 700:
    if total_length > 375:
        pair = doc_couples[0]
        half_length = doc_len // 2

        if pair[0] <= half_length and pair[1] <= half_length:
            # Take the first half of the document
            doc_len = half_length
            clauses = clauses[: half_length]
            emotion_categorys = emotion_categorys[: half_length]
            emotion_tokens = emotion_tokens[: half_length]

        else:
            # Take the second half of the document
            doc_len = doc_len - half_length
            for i in range(len(doc_couples)):
                doc_couples[i][0] -= half_length
                doc_couples[i][1] -= half_length
            clauses = clauses[half_length:]
            emotion_categorys = emotion_categorys[half_length:]
            emotion_tokens = emotion_tokens[half_length:]

    assert doc_len == len(clauses) == len(emotion_categorys) == len(emotion_tokens)

    return doc_len, doc_couples, clauses, emotion_categorys, emotion_tokens


class StoreProcessedData(object):
    """Stores the processed data for each document"""

    def __init__(self, doc_id, doc_len, text, clause_list, clause_len_list, f_query_len_list, b_query_len_list,
                 emotion_category_list, emotion_token_list, pairs,
                 forward_query_list, backward_query_list, sentiment_query_list,
                 f_e_query_answer, f_c_query_answer_list, b_c_query_answer, b_e_query_answer_list,
                 sentiment_answer_list, forward_query_seg=None, backward_query_seg=None):
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.text = text
        self.clause_list = clause_list
        self.clause_len_list = clause_len_list
        self.emotion_category_list = emotion_category_list
        self.emotion_token_list = emotion_token_list
        self.pairs = pairs

        self.f_query_list = forward_query_list
        self.f_query_len_list = f_query_len_list
        self.f_e_query_answer = f_e_query_answer
        self.f_c_query_answer = f_c_query_answer_list
        self.f_query_seg = forward_query_seg

        self.b_query_list = backward_query_list
        self.b_query_len_list = b_query_len_list
        self.b_c_query_answer = b_c_query_answer
        self.b_e_query_answer = b_e_query_answer_list
        self.b_query_seg = backward_query_seg

        self.sentiment_query = sentiment_query_list
        self.sentiment_answer = sentiment_answer_list


def processData():
    configs = Config()
    dataset_name_list = []
    for i in range(1, 11):
        dataset_name_list.append('fold{}'.format(i))

    dataset_type_list = ['train', 'test']
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            output_path = 'ecpe_extractor/data/CV/' + dataset_name + '_' + dataset_type + '_dual.pt'
            input_path = 'ecpe_extractor/data/CV/' + dataset_name + '_' + dataset_type + '.json'

            # output_path = 'ecpe_extractor/dataEnglish/CV/' + dataset_name + '_' + dataset_type + '_dual.pt'
            # input_path = 'ecpe_extractor/dataEnglish/CV/' + dataset_name + '_' + dataset_type + '.json'
            sample_list = []

            # Raw data
            with open(input_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)

            # For each document
            for doc in dataset:
                doc_id = int(doc['doc_id'])  # ID
                doc_len = int(doc['doc_len'])  # Length
                doc_couples = doc['pairs']  # ECPE
                doc_clauses = doc['clauses']  # Clause

                # print("Document ID:",doc_id)
                # print("Document pairs:",doc_couples)

                clause_list = []
                emotion_categorys = []
                emotion_tokens = []

                for i in range(len(doc_clauses)):
                    # print(doc_clauses[i])
                    clause_list.append(doc_clauses[i]['clause'])
                    emotion_category = doc_clauses[i]['emotion_category']
                    # print(doc_clauses[i])
                    # print(emotion_category)
                    if '&' in emotion_category:
                        emotion_category = emotion_category.split('&')[0]
                    emotion_categorys.append(emotion_category)
                    emotion_tokens.append(doc_clauses[i]['emotion_token'])

                # Truncate the document
                doc_len, doc_couples, clause_list, emotion_categorys, emotion_tokens = \
                    truncate_document(doc_len, doc_couples, clause_list, emotion_categorys, emotion_tokens)

                # List for emotion and cause
                emotion_list, cause_list = zip(*doc_couples)

                emotion_list = list(set(emotion_list))

                cause_list = list(set(cause_list))

                # Calculate the length for each clause
                clause_len_list = [len(clause) for clause in clause_list]
                text = ''.join(clause_list)

                # Initialize query lists and answer lists
                forward_query_list = [["是描述事件原因的原因子句"]]
                backward_query_list = [["是描述事件情感的情感子句"]]

                # forward_query_list = [["is a causal subordinate clause describing the cause of an event."]]
                # backward_query_list = [["is an emotional subordinate clause describing the emotion of an event."]]

                f_e_query_answer = [[0] * doc_len]
                b_c_query_answer = [[0] * doc_len]
                b_e_query_answer = []
                f_c_query_answer = []
                f_query_len_list = [8]
                b_query_len_list = [8]
                sentiment_query_list = []
                sentiment_answer_list = []

                for emo_idx in emotion_list:
                    # print(f_e_query_answer)
                    # print(f_e_query_answer[0])
                    # print(f_e_query_answer[0][emo_idx - 1])

                    f_e_query_answer[0][emo_idx - 1] = 1
                    sc = configs.sentiment_category[emotion_categorys[emo_idx - 1]]
                    sentiment_answer_list.append(sc)
                for cau_idx in cause_list:
                    b_c_query_answer[0][cau_idx - 1] = 1

                # Handle queries and answers
                temp_emotion = set()
                for pair in doc_couples:
                    emotion_idx = pair[0]
                    cause_idx = pair[1]
                    if emotion_idx not in temp_emotion:
                        causes = []
                        for e, c in doc_couples:
                            if e == emotion_idx:
                                causes.append(c)
                        query_f = clause_list[emotion_idx - 1] + '发生的原因是'
                        # query_f = clause_list[emotion_idx - 1] + 'The reason for this happening is'
                        forward_query_list.append([query_f])
                        f_query_len_list.append(len(query_f))
                        f_query2_answer = [0] * doc_len
                        for c_idx in causes:
                            f_query2_answer[c_idx - 1] = 1
                        f_c_query_answer.append(f_query2_answer)
                        temp_emotion.add(emotion_idx)
                    query_b = clause_list[cause_idx - 1] + '导致了'
                    # query_b = clause_list[cause_idx - 1] + 'leads to'
                    backward_query_list.append([query_b])
                    b_query_len_list.append(len(query_b))
                    b_query2_answer = [0] * doc_len
                    b_query2_answer[emotion_idx - 1] = 1
                    b_e_query_answer.append(b_query2_answer)

                # Create a StoreProcessedData object and add it to the sample list
                temp_sample = StoreProcessedData(doc_id, doc_len, text, clause_list, clause_len_list, f_query_len_list,
                                                 b_query_len_list, emotion_categorys, emotion_tokens, doc_couples,
                                                 forward_query_list, backward_query_list, sentiment_query_list,
                                                 f_e_query_answer, f_c_query_answer, b_c_query_answer, b_e_query_answer,
                                                 sentiment_answer_list, None, None)
                sample_list.append(temp_sample)

            # Save the sample list to the output path
            torch.save(sample_list, output_path)
            print(f'Processing Document - Output Path: {output_path} - Finished.')
