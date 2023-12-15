import torch
from torch.utils.data import Dataset, DataLoader


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data['_docid_list']
        self.clause_list = pre_data['_clause_list']
        self.doc_len_list = pre_data['_doc_len_list']
        self.clause_len_list = pre_data['_clause_len_list']
        self.pairs = pre_data['_pairs']

        self._f_emo_query = pre_data['_f_emo_query']
        self._f_cau_query = pre_data['_f_cau_query']
        self._f_emo_query_len = pre_data['_f_emo_query_len']
        self._f_cau_query_len = pre_data['_f_cau_query_len']
        self._f_emo_query_answer = pre_data['_f_emo_query_answer']
        self._f_cau_query_answer = pre_data['_f_cau_query_answer']
        self._f_emo_query_mask = pre_data['_f_emo_query_mask']
        self._f_cau_query_mask = pre_data['_f_cau_query_mask']
        self._f_emo_query_seg = pre_data['_f_emo_query_seg']
        self._f_cau_query_seg = pre_data['_f_cau_query_seg']

        self._b_emo_query = pre_data['_b_emo_query']
        self._b_cau_query = pre_data['_b_cau_query']
        self._b_emo_query_len = pre_data['_b_emo_query_len']
        self._b_cau_query_len = pre_data['_b_cau_query_len']
        self._b_emo_query_answer = pre_data['_b_emo_query_answer']
        self._b_cau_query_answer = pre_data['_b_cau_query_answer']
        self._b_emo_query_mask = pre_data['_b_emo_query_mask']
        self._b_cau_query_mask = pre_data['_b_cau_query_mask']
        self._b_emo_query_seg = pre_data['_b_emo_query_seg']
        self._b_cau_query_seg = pre_data['_b_cau_query_seg']

        self._forward_c_num = pre_data['_forward_c_num']
        self._backward_e_num = pre_data['_backward_e_num']


def calculate_metrics(pred_emotions, true_emotions, pred_causes, true_causes, pred_pairs, true_pairs, document_length):
    true_positives_emo, false_positives_emo, false_negatives_emo = 0, 0, 0
    true_positives_cau, false_positives_cau, false_negatives_cau = 0, 0, 0
    true_positives_pair, false_positives_pair, false_negatives_pair = 0, 0, 0

    for i in range(1, document_length + 1):
        # Calculate metrics for predicted emotions
        if i in pred_emotions and i in true_emotions:
            true_positives_emo += 1
        elif i in pred_emotions and i not in true_emotions:
            false_positives_emo += 1
        elif i not in pred_emotions and i in true_emotions:
            false_negatives_emo += 1

        # Calculate metrics for predicted causes
        if i in pred_causes and i in true_causes:
            true_positives_cau += 1
        elif i in pred_causes and i not in true_causes:
            false_positives_cau += 1
        elif i not in pred_causes and i in true_causes:
            false_negatives_cau += 1

    for pred_pair in pred_pairs:
        # Calculate metrics for predicted pairs
        if pred_pair in true_pairs:
            true_positives_pair += 1
        else:
            false_positives_pair += 1

    for true_pair in true_pairs:
        if true_pair not in pred_pairs:
            false_negatives_pair += 1

    return [true_positives_emo, false_positives_emo, false_negatives_emo], [true_positives_cau, false_positives_cau,
                                                                            false_negatives_cau], [true_positives_pair,
                                                                                                   false_positives_pair,
                                                                                                   false_negatives_pair]


def calculate_evaluation_metrics(metrics):
    true_positives, false_positives, false_negatives = metrics[0], metrics[1], metrics[2]
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return [f1_score, precision, recall]


def extract_emotion_cause_pairs(batch, model, tokenizer, emo_dictionary):
    """
    Extract emotion-cause pairs from a batch of data.
    Args:
        batch: A batch of data containing clauses and pairs.
        model: The ECPE model.
        tokenizer: The BERT tokenizer.
        emo_dictionary: A dictionary containing emotion-related terms.

    Some of returns:
        pred_emo_final: Predicted emotion indices.
        pred_cau_final: Predicted cause indices.
        pred_pair_final: Predicted emotion-cause pairs.
    """
    # Extract data from the batch
    docid_list, clause_list, pairs, \
        feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
        fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
        bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
        beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask \
        = batch
    doc_id, clause_list, true_pairs = docid_list[0], clause_list[0], pairs[0]
    true_emo, true_cau = zip(*true_pairs)
    true_emo, true_cau = list(true_emo), list(true_cau)
    document = ''.join(clause_list)
    document = ' '.join(document).split(' ')

    predicted_emotions = []
    pred_pair_f = []
    pred_pair_f_probs = []
    pred_pair_b = []
    pred_pair_b_pro = []
    pred_emo_single = []
    pred_cau_single = []
    intermediate_results = []
    intermediate_results_probs = []

    # Predict emotions
    f_emo_pred = model(feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj)
    f_emo_pred = f_emo_pred.cpu().squeeze()
    temp_emo_f_prob = f_emo_pred.masked_select(fe_an_mask.bool()).cpu().numpy().tolist()

    for idx in range(len(temp_emo_f_prob)):
        # if temp_emo_f_prob[idx] > 0.99 or (temp_emo_f_prob[idx] > 0.5 and idx + 1 in emo_dictionary[str(doc_id)]):
        if temp_emo_f_prob[idx] > 0.99 or (temp_emo_f_prob[idx] > 0.5):
            predicted_emotions.append(idx)
            pred_emo_single.append(idx + 1)

    # Predict causes
    for idx_emo in predicted_emotions:
        f_query = clause_list[idx_emo] + 'The cause of event happening is:'
        f_query = ' '.join(f_query).split(' ')
        f_qa = ['[CLS]'] + f_query + ['[SEP]'] + document
        f_qa = tokenizer.convert_tokens_to_ids([w.lower() if w not in ['[CLS]', '[SEP]'] else w for w in f_qa])
        f_mask = [1] * len(f_qa)
        f_seg = [0] * (len(f_query) + 2) + [1] * len(document)
        f_len = len(f_query)
        f_qa = torch.LongTensor([f_qa])
        f_mask = torch.LongTensor([f_mask])
        f_seg = torch.LongTensor([f_seg])
        f_len = [f_len]
        f_clause_len = fe_clause_len
        f_doc_len = fe_doc_len
        f_adj = fe_adj
        f_cau_pred = model(f_qa, f_mask, f_seg, f_len, f_clause_len, f_doc_len, f_adj)
        temp_cau_f_prob = f_cau_pred[0].cpu().numpy().tolist()

        for idx_cau in range(len(temp_cau_f_prob)):
            if temp_cau_f_prob[idx_cau] > 0.5 and abs(idx_emo - idx_cau) <= 11:
                if idx_cau + 1 not in pred_cau_single:
                    pred_cau_single.append(idx_cau + 1)
                prob_t = temp_emo_f_prob[idx_emo] * temp_cau_f_prob[idx_cau]
                if idx_cau - idx_emo >= 0 and idx_cau - idx_emo <= 2:
                    pass
                else:
                    prob_t *= 0.9
                pred_pair_f_probs.append(prob_t)
                pred_pair_f.append([idx_emo + 1, idx_cau + 1])
                intermediate_results.append([idx_emo + 1, idx_cau + 1])
                intermediate_results_probs.append(prob_t)

    pred_emo_final = []
    pred_cau_final = []
    pred_pair_final = []

    for i, pair in enumerate(pred_pair_f):
        if pred_pair_f_probs[i] > 0.5:
            pred_pair_final.append(pair)

    for pair in pred_pair_final:
        if pair[0] not in pred_emo_final:
            pred_emo_final.append(pair[0])
        if pair[1] not in pred_cau_final:
            pred_cau_final.append(pair[1])

    return pred_emo_final, pred_cau_final, pred_pair_final, true_emo, true_cau, pred_emo_single, pred_cau_single, true_pairs, clause_list


def evaluate_one_batch(batch, model, tokenizer, emo_dictionary):
    """
    Evaluate one batch of data for emotion-cause pair extraction.
    Args:
        batch: A batch of data containing clauses and pairs.
        model: The ECPE model.
        tokenizer: The BERT tokenizer.
        emo_dictionary: A dictionary containing emotion-related terms.

    Returns:
        Metrics for emotion prediction, cause prediction, and pair prediction.
    """
    pred_emo_final, pred_cau_final, pred_pair_final, true_emo, true_cau, pred_emo_single, pred_cau_single, true_pairs, clause_list = extract_emotion_cause_pairs(
        batch, model, tokenizer,
        emo_dictionary)

    # Calculate metrics using calculate_metrics function
    metric_e_s, metric_c_s, _ = calculate_metrics(pred_emo_single, true_emo, pred_cau_single, true_cau, pred_pair_final,
                                                  true_pairs, len(clause_list))
    metric_e, metric_c, metric_p = calculate_metrics(pred_emo_final, true_emo, pred_cau_final, true_cau,
                                                     pred_pair_final,
                                                     true_pairs, len(clause_list))

    return metric_e, metric_c, metric_p, metric_e_s, metric_c_s


def evaluate(test_loader, model, tokenizer, emo_dictionary):
    """
    Evaluate the emotion-cause pair extraction model on the test data.
    Args:
        configs: Configuration parameters.
        test_loader: Data loader for the test data.
        model: The ECPE model.
        tokenizer: The BERT tokenizer.
        emo_dictionary: A dictionary containing emotion-related terms.

    Returns:
        Evaluation metrics for emotion prediction, cause prediction, and pair prediction.
    """
    model.eval()
    all_emo, all_cau, all_pair = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    all_emo_s, all_cau_s = [0, 0, 0], [0, 0, 0]

    for batch in test_loader:
        emo, cau, pair, emo_s, cau_s = evaluate_one_batch(batch, model, tokenizer, emo_dictionary)
        for i in range(3):
            all_emo[i] += emo[i]
            all_cau[i] += cau[i]
            all_pair[i] += pair[i]
            all_emo_s[i] += emo_s[i]
            all_cau_s[i] += cau_s[i]

    eval_emo = calculate_evaluation_metrics(all_emo)
    eval_cau = calculate_evaluation_metrics(all_cau)
    eval_pair = calculate_evaluation_metrics(all_pair)
    eval_emo_s = calculate_evaluation_metrics(all_emo_s)
    eval_cau_s = calculate_evaluation_metrics(all_cau_s)

    return eval_emo, eval_cau, eval_pair, eval_emo_s, eval_cau_s
