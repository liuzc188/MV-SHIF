# @Time : 2023/11/15 15:24
# @Author : Cheng Yang
# @File ：txtToJson.py

import json
import re


def convert_to_json(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    i = 0
    doc_lengths = []

    while i < len(lines):
        doc_info = lines[i].split()
        doc_id = doc_info[0]
        doc_len = int(doc_info[1])
        doc_lengths.append(doc_len)  # 统计所有文档的平均长度

        pairs = eval(lines[i + 1])

        all_clause_len = []
        clauses = []
        for j in range(i + 2, i + 2 + doc_len):
            clause_info = lines[j].split(',')
            clause_id = clause_info[0]
            emotion_category = clause_info[1]
            emotion_token = clause_info[2]
            clause = ','.join(clause_info[3:])

            clause_data = {
                "clause_id": clause_id,
                "emotion_category": emotion_category if emotion_category != 'null' else "null",
                "emotion_token": emotion_token if emotion_token != 'null' else "null",
                "clause": clause
            }
            clauses.append(clause_data)
            all_clause_len.append(len(clause))
        average_len = sum(all_clause_len)
        print(average_len)

        document = {
            "doc_id": doc_id,
            "doc_len": doc_len,
            "pairs": pairs,
            "clauses": clauses
        }
        data.append(document)

        i += 2 + doc_len

    total_length = sum(doc_lengths)
    average_length = total_length / len(doc_lengths) if len(doc_lengths) > 0 else 0
    print("average_length", average_length)

    with open(output_file_path, 'w') as out_file:
        json.dump(data, out_file, indent=4)


# for data_type in ['train', 'val','test']:
#
#     for i in range(1, 11):
#         # 读取原始数据
#         input_file_path = f'./Raw/fold{i}_{data_type}.txt'
#
#         # 保存JSON数据到json文件
#         output_file_path = f'./TVT/fold{i}_{data_type}.json'  # 替换为你想要保存的文件路径
#
#         convert_to_json(input_file_path, output_file_path)
#
#         print(f"数据已成功转换并保存到 {output_file_path}")

input_file_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/data/all_data_pair.txt'
output_file_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/data/all_data_pair.json'
convert_to_json(input_file_path, output_file_path)
print(f"数据已成功转换并保存到 {output_file_path}")