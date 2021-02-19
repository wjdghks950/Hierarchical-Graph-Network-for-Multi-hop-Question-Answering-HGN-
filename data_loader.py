import os
import copy
import logging
import itertools
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset
import dgl

logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class HotpotExample(object):
    """
    A single train/test example for fine-tuning BERT on paragraph selection (sequence classification).

    Args:
        sup_facts: Supporting facts
        question: Question
        context: Context (list of documents in `List[title, List[sentence1, sentence2, ...]]`)
        answer: Answer text
        level: Difficulty level of the question
        _id: _id of QA pair
        _type: type of QA pair (e.g., bridge, comparison)
    """

    def __init__(self, sup_facts, question, context, answer, level, _id, _type):
        self.sup_facts = sup_facts
        self.question = question
        self.context = context
        self.answer = answer
        self.level = level
        self._id = _id
        self._type = _type

    def __repr__(self):
        delineate = "#" * 50
        s = "\nQuestion: {}\nAnswer: {}\n".format(self.question, self.answer)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class HotpotFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, node_indices, span_dict, para_lbl, sent_lbl, span_idx, answer_type_lbl):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.node_indices = node_indices  # `node_idx`: Dict[Dict[List]]: {Para_idx1: {sent_idx1: {ent_idx1, ent_idx2, ...}, sent_idx2: {...}}, Para_idx2: {...}}
        self.span_dict = span_dict  # `span_dict`: Dict[List[Tuples]]: start_idx & end_idx for each paragraph, sentence, and entity ({"paragraph":[(p1_start, p1_end)], "sentence":[(s1_start, s1_end), (...)]})
        self.para_lbl = para_lbl
        self.sent_lbl = sent_lbl
        self.span_idx = span_idx  # (start_idx, end_idx)
        self.answer_type_lbl = answer_type_lbl  # `answer_type_lbl`: [span, entity, yes/no]

    def __repr__(self):
        delineate = "#" * 50
        s = "\n[Question; Context]: {}\nAttention_mask: {}\nAnswer_type: {}\n".format(self.input_ids, self.attention_mask, self.answer_type_lbl)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class ParagraphSelectorFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        delineate = "#" * 50
        s = "\n[Question; Context]: {}\n".format(self.input_ids)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class HotpotProcessor(object):
    """Processor for retrieving relevant paragraphs from the HotpotQA data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """ Read the Deceptive Opinion Spam File"""
        if os.path.exists(input_file):
            df = pd.read_json(input_file)
        else:
            raise Exception("Error: {} does not exist".format(input_file))
        return df

    def _create_examples(self, df, mode):
        """Creates examples for the training and dev sets."""
        print("({}) Creating ParaSelect examples...".format(mode))
        examples = []
        for i, row in df.iterrows():
            sup_facts = row["supporting_facts"]
            question = row["question"]
            context = row["context"]
            answer = row["answer"]
            level = row["level"]
            _id = row["_id"]
            _type = row["type"]

            if i % 1000 == 0:
                logger.info(row)
            examples.append(HotpotExample(sup_facts=sup_facts, question=question, context=context, answer=answer, level=level, _id=_id, _type=_type))

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file

        print("File to read >> {}\n".format(file_to_read))

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "para_select": HotpotProcessor,
    "train_model": HotpotProcessor,  # Distractor setting
    # TODO: Need another processor for `Full-wiki setting`
}


def convert_examples_to_features(args, examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    try:
        from predict import ParaPredictor  # TODO: Need to move this to outside of the for-loop (being initialized repeatedly)
        para_ranker = ParaPredictor(args, tokenizer)
    except ImportError:
        raise Exception("Error: ParaPredictor cannot be imported from predict.py")

    features = []
    print("Task >> ({})".format(args.task))
    # print("Examples (length) >> ({})".format(len(examples)))
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if args.task == "para_select":  # sup_facts, question, context, answer, level, _id, _type
            sup_titles, sent_idx = zip(*example.sup_facts)
            ctx_titles, contexts = zip(*example.context)
            label_list = [1 if title in sup_titles else 0 for title in ctx_titles]
            context_list = [''.join(ctx) for ctx in contexts]

            for i in range(len(label_list)):
                question_tokens = tokenizer.tokenize(example.question)
                context_tokens = tokenizer.tokenize(context_list[i])

                tokens = question_tokens + [sep_token] + context_tokens

                # Account for [CLS] and [SEP]
                special_tokens_count = 2
                if len(tokens) > max_seq_len - special_tokens_count:
                    tokens = tokens[:(max_seq_len - special_tokens_count)]

                # Add [SEP] token
                tokens += [sep_token]
                token_type_a_ids = [sequence_a_segment_id] * (len(question_tokens) + 1)
                token_type_b_ids = [sequence_b_segment_id] * (len(context_tokens) + 1)

                # Add [CLS] token
                tokens = [cls_token] + tokens
                token_type_a_ids = [cls_token_segment_id] + token_type_a_ids
                token_type_ids = token_type_a_ids + token_type_b_ids
                if len(token_type_ids) > max_seq_len:
                    token_type_ids = token_type_ids[:max_seq_len]

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_len - len(input_ids)
                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
                assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
                assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

                # if i % 10000 == 0:
                #     logger.info("*** Example ***")
                #     logger.info("Question: %s" % example.question)
                #     logger.info("Context: %s" % context_list[i])
                #     logger.info("Concatenated [Question; Context] %s" % tokens)
                #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                #     logger.info("label: %s (id = %s) (Gold == 1 / Not Gold == 0)" % (str(label_list[i]), example._id))
                #     exit()

                features.append(
                    ParagraphSelectorFeatures(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label_id=label_list[i]
                    ))

        else:  # args.task == "train_model"
            nlp = spacy.load("en_core_web_sm")
            # (i) Retrieve ``First-hop`` paragraphs
            para_selected_idx = para_ranker.title_matching(example)
            num_para = len(para_selected_idx)
            if num_para > 2:  # Only two paragraphs with the highest ranking scores are selected
                para_scores = para_ranker.para_predict(example)
                para_selected_scores = para_scores[para_selected_idx]
                para_selected_dict = dict(zip(para_selected_idx, para_selected_scores))
                para_ordered = {k: v for k, v in sorted(para_selected_dict.items(), key=lambda item: item[1], reverse=True)}
                para_idx, _ = map(list, zip(*para_ordered.items()))
                para_list = para_idx[:2]

            elif num_para < 2:  # Find entities within the paragraph that overlap with the ones in the question
                ctx_titles, contexts = zip(*example.context)
                ent_para_idx = []
                for i in range(len(ctx_titles)):
                    para_string = ''.join(contexts[i])
                    doc = nlp(para_string)
                    entities = [e.text.lower() for e in doc.ents]
                    if any(entity in example.question.lower() for entity in entities):
                        ent_para_idx.append(i)
                if num_para + len(ent_para_idx) == 2:
                    para_selected_idx.extend(ent_para_idx)
                elif num_para + len(ent_para_idx) > 2:
                    para_selected_idx.extend(ent_para_idx)
                    para_selected_idx = list(set(para_selected_idx))
                    para_scores = para_ranker.para_predict(example)
                    para_selected_scores = para_scores[para_selected_idx]
                    para_selected_dict = dict(zip(para_selected_idx, para_selected_scores))
                    para_ordered = {k: v for k, v in sorted(para_selected_dict.items(), key=lambda item: item[1], reverse=True)}
                    para_idx, _ = map(list, zip(*para_ordered.items()))
                    para_list = para_idx[:2]
                else:  # num_para == 0
                    para_scores = para_ranker.para_predict(example)
                    top_n = 2
                    para_list = np.argsort(para_scores)[::-1][:top_n]

            else:  # num_para == 2
                para_list = para_selected_idx

            # (ii) Select the ``Second-hop`` paragraphs
            # TODO: Retrieve the second-hop paragraphs without using hyperlinks.
            para_scores = para_ranker.para_predict(example)
            N = 4  # Top-N paragraphs
            para_list = np.argsort(para_scores)[::-1][:N]  # TODO: Tentative measure for selecting paragraphs (Select top-N paragraphs)

            # Concatenate all the selected paragraphs to construct `C` (i.e., the context)
            context_titles, context_list = zip(*example.context)
            context_titles_selected = np.array(context_titles)[para_list].tolist()
            context_selected = np.array(context_list)[para_list].tolist()

            # node_idx: the node_idx for paragraph, sentence and entities (for graph construction)
            # *_span_list: the start_idx & end_idx for paragraphs, sentences and entities for (node initialization)
            node_idx = {}
            para_span_list = []
            sent_span_list = []
            ent_span_list = []
            all_ent_list = []
            sent_idx = 0
            ent_idx = 0
            question_start = 0
            para_start = 0

            question_tokens = tokenizer.tokenize(example.question)
            question_span = [(question_start, len(question_tokens) + 2)]  # `2` accounts for [CLS] and [SEP] appended later

            # Assign `node_idx` and extract `spans` of paragraph, sentence and entity
            for i, para in enumerate(context_selected):
                node_idx[i] = defaultdict(list)
                para_seq = tokenizer.tokenize(''.join(para))
                para_span_list.append((para_start, para_start + len(para_seq)))
                para_start += len(para_seq)
                sent_start = 0
                for sent in para:
                    try:
                        doc = nlp(sent)
                    except TypeError as e:
                        print("Error: {} / Sent: {}".format(e, sent))
                    ent_list = list(set([e.text.lower() for e in doc.ents]))
                    all_ent_list.extend(ent_list)
                    sent_seq = tokenizer.tokenize(sent)
                    sent_span_list.append((sent_start, sent_start + len(sent_seq)))
                    for ent in ent_list:
                        node_idx[i][sent_idx].append(ent_idx)  # Entity index list (within j_th sentence)
                        ent_tok = tokenizer.tokenize(ent)
                        for k in range(len(sent_seq) - len(ent_tok)):
                            if sent_seq[k : k + len(ent_tok)] == ent_tok:
                                ent_span_list.append((sent_start + k, sent_start + k + len(ent_tok)))  # Entity span found
                                break
                        ent_idx += 1
                    sent_idx += 1
                    sent_start += len(sent_seq)

            assert len(all_ent_list) == ent_idx

            all_ent_dict = dict(zip(all_ent_list, list(range(ent_idx))))  # `all_ent_dict` to assign ent_idx to question node
            q_idx = 0
            q2ent = defaultdict(list)
            for entity in all_ent_list:
                if entity in example.question.lower():
                    q2ent[q_idx].append(all_ent_dict[entity])
            
            span_dict = {"question": question_span, "paragraph": para_span_list, "sentence": sent_span_list, "entity": ent_span_list}
            node_indices = (q2ent, node_idx)
            
            # print("Question span : {}".format(question_span))
            # print("Paragraph span : {}".format(para_span_list))
            # print("Sentence span : {}".format(sent_span_list))
            # print("Entity span : {}".format(ent_span_list))
            # print("Node_idx : {}".format(node_idx))
            # print("q2ent: {}".format(q2ent))
            # print("Paragraph sequence: {}" .format(para_seq[para_span_list[0][0]:para_span_list[0][1]]))
            # print("Entities: {}" .format(sent_seq[ent_span_list[0][0]:ent_span_list[0][1]]))

            para_lbl = []
            sup_titles, sent_idx = map(np.array, zip(*example.sup_facts))
            para_lbl = [1 if title in sup_titles else 0 for title in context_titles_selected]  # Construct labels for paragraph
            if len(para_lbl) < args.num_paragraphs:  # num_paragraphs == 4
                para_lbl += [0] * (args.num_paragraphs - len(para_lbl))
            else:
                para_lbl = para_lbl[:args.num_paragraphs]

            sent_lbl = []
            for c_i, context in enumerate(context_selected):
                temp_lbl = np.zeros(len(context), dtype=int)
                if context_titles_selected[c_i] in sup_titles:
                    sup_para_idx = np.where(context_titles_selected[c_i] == sup_titles)[0].tolist()
                    sup_fact_idx = sent_idx[sup_para_idx]
                    real_sup_fact_idx = sup_fact_idx <= len(context) - 1  # For erroneous sup_fact index cases
                    temp_lbl[sup_fact_idx[real_sup_fact_idx]] = 1
                if c_i == 0:
                    sent_lbl = temp_lbl.copy()
                else:  # c_i > 0
                    sent_lbl = np.append(sent_lbl, temp_lbl)

            if type(sent_lbl).__module__ == np.__name__:
                sent_lbl = sent_lbl.tolist()
                
            if len(sent_lbl) < args.num_sentences:  # num_sentences = 40
                sent_lbl += [0] * (args.num_sentences - len(sent_lbl))
            else:
                sent_lbl = sent_lbl[:args.num_sentences]

            assert len(para_span_list) <= N
            assert args.num_sentences == len(sent_lbl)
            assert args.num_paragraphs == len(para_lbl)
            # TODO: Need ent_lbl ("...candidate entities include all entities in the question and those that match the titles in the context.")

            context = ' '.join([''.join(c) for c in context_selected])

            # `answer_type_lbl`: [span, entity, yes/no]
            # `span_idx` for answer within the given context of length "n"
            if (example.answer.lower() in ["yes", "no"]) and (example._type == "comparison"):
                answer_type_lbl = 2  # Yes/no type
            else:
                if (example.answer in context) and (example.answer.lower() in all_ent_list):
                    answer_type_lbl = 1  # Entity type
                else:
                    answer_type_lbl = 0  # Span type
                answer_tok = tokenizer.tokenize(example.answer)
                context_tok = tokenizer.tokenize(context)
                for sub_i in range(len(context_tok) - len(answer_tok)):
                    if context_tok[sub_i : sub_i + len(answer_tok)] == answer_tok:
                        span_idx = (sub_i, sub_i + len(answer_tok))  # The answer `span_idx` (ground-truth)

            print("node_idx: ", node_idx)
            print("span_idx[entity]: ", len(span_dict["entity"]))
            # print("Question: {}".format(example.question))
            # print("Answer: {}".format(example.answer))
            # print("Answer type: {} => {}".format(example._type, answer_type_lbl))
            # print("Para_lbl: {}".format(para_lbl))
            # print("Sent_lbl: {}".format(sent_lbl))
            # print("Span_idx: {}".format(span_idx))
            # print("All Entities: {}".format(all_ent_list))
            # print("Context: {}".format(context_tok))
            # print("Answer Span: {}".format(context_tok[span_idx[0]: span_idx[1]]))
            # print("Supporting sentences: {}".format(example.sup_facts))

            context_tokens = tokenizer.tokenize(context)
            tokens = question_tokens + [sep_token] + context_tokens
            
            # Account for [CLS] and [SEP] ([SEP] for the last index in the sequence)
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_a_ids = [sequence_a_segment_id] * (len(question_tokens) + 1)
            token_type_b_ids = [sequence_b_segment_id] * (len(tokens) - len(token_type_a_ids))

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_a_ids = [cls_token_segment_id] + token_type_a_ids
            token_type_ids = token_type_a_ids + token_type_b_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

            if ex_index < 2:
                logger.info("*** Example ***")
                logger.info("Question: %s" % example.question)
                logger.info("Context: %s" % context_list[i])
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("Answer: %s" % example.answer)
                logger.info("Supporting Facts: %s" % example.sup_facts)
                logger.info("Answer Type: %s" % example._type)
                logger.info("Node_idx: {}".format(node_idx))

            features.append(
                HotpotFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    node_indices=node_indices,
                    span_dict=span_dict,
                    para_lbl=para_lbl,
                    sent_lbl=sent_lbl,
                    span_idx=span_idx,
                    answer_type_lbl=answer_type_lbl
                ))

    return features


def graph_constructor(args, node_indices, span_dict):
    '''
    Args
    
    args : Model argument
    node_indices : Tuple of index dicts for question, paragraph, sentence and entity nodes
    span_dict : Span for paragraph, sentence and entity nodes

    ----------

    node_idx : Dict[Dict[List]]: {Para_idx1: {sent_idx1: {ent_idx1, ent_idx2, ...}, sent_idx2: {...}}, Para_idx2: {...}}

    '''
    q2ent, node_idx = node_indices
    # The edges should be bi-directional
    ps = ("paragraph", "ps", "sentence")
    sp = ("sentence", "sp", "paragraph")
    se = ("sentence", "se", "entity")
    es = ("entity", "es", "sentence")
    pp = ("paragraph", "pp", "paragraph")
    ss = ("sentence", "ss", "sentence")
    qp = ("question", "qp", "paragraph")  # Max = 2 paragraphs (two-hop)
    pq = ("paragraph", "pq", "question")
    qe = ("question", "qe", "entity")
    eq = ("entity", "eq", "question")  # TODO: Entities within sentences could exist within the question!

    # print("<================================== Constructing dgl.heterograph ... ==================================>")
    # print("Node_idx: ", node_idx)
    data_dict = {}
    p2s = []
    s2e = []
    q2p = []
    p2p = []
    s2s = []
    q2e = []

    para_idx_list, sent2ent = map(list, zip(*node_idx.items()))
    question_idx = 0
    question_idx_list = [question_idx] * len(para_idx_list)
    q2p += list(zip(question_idx_list, para_idx_list))
    p2p += list(itertools.permutations(para_idx_list, r=2))

    for i in range(len(para_idx_list)):
        para_idx = [para_idx_list[i]] * len(sent2ent[i])
        p2s += list(zip(para_idx, list(sent2ent[i].keys())))
        s2e += [(s, e) for s, e_list in sent2ent[i].items() for e in e_list]
        s_idx = list(sent2ent[i].keys())
        s2s += [(s_idx[i], s_idx[i + 1]) for i in range(len(s_idx) - 1)]
    
    print("node_idx: ", node_idx)
    print("span_dict: ", len(span_dict["entity"]))

    if len(q2ent) > 0:
        q_idx = list(q2ent.keys())
        ent_list = list(q2ent.values())[0]
        q_idx_list = q_idx * len(ent_list)
        q2e += list(zip(q_idx_list, ent_list))
    
    else:  # No entity within the question
        pass  # TODO: Need to handle the case where there is no entity present within the question, yet "qe" and "eq" should exist.

    data_dict[ps] = p2s
    data_dict[sp] = [t[::-1] for t in p2s]
    data_dict[se] = s2e
    data_dict[es] = [t[::-1] for t in s2e]
    data_dict[qp] = q2p
    data_dict[pq] = [t[::-1] for t in q2p]
    data_dict[pp] = p2p
    data_dict[ss] = s2s
    data_dict[ss] += [t[::-1] for t in s2s]
    data_dict[qe] = q2e
    data_dict[eq] = [t[::-1] for t in q2e]

    g = dgl.heterograph(data_dict, device=device)
    graph_out = (g, node_idx, span_dict)

    print("Num entities (nodes vs. spans): {} / {}".format(len(g.nodes("entity")), len(span_dict["entity"])))

    assert len(para_idx_list) == len(span_dict["paragraph"])
    assert len(g.nodes("entity")) == len(span_dict["entity"])

    # print("data_dict: ", data_dict)
    # print("*** g *** : ", g)
    # print("Graph constructor!")

    return graph_out


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        else:
            raise Exception("For mode, Only train, dev is available")

        print("Length of example ({}): {}".format(mode, len(examples)))
        features = convert_examples_to_features(args, examples, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s / Length >> (%d)", cached_features_file, len(features))
        torch.save(features, cached_features_file)

    if args.task == "para_select":
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids)

        print("*********Dataset preprocessing complete**********")
        return dataset
        
    elif args.task == "train_model":
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_question_ends = torch.tensor([f.input_ids.index(tokenizer.sep_token_id) for f in features])
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        for i, f in enumerate(features[:2]):
            print("input_id >> ", f.input_ids)
            print("question_ends >> ", all_question_ends[i])
            print("para_lbl >> ", f.para_lbl)
            print("sent_lbl >> ", f.sent_lbl)
            print("Span_dict >> ", f.span_dict)
            print("q2ent >> ", f.node_indices[0])
            print("node_idx >> ", f.node_indices[1])

        all_para_lbl = torch.tensor([f.para_lbl for f in features], dtype=torch.long)
        all_sent_lbl = torch.tensor([f.sent_lbl for f in features], dtype=torch.long)
        all_answer_type_lbl = torch.tensor([f.answer_type_lbl for f in features], dtype=torch.long)
        all_span_idx = torch.tensor([f.span_idx for f in features], dtype=torch.long)
        all_graph_out = [graph_constructor(args, f.node_indices, f.span_dict) for f in features]

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_para_lbl, all_sent_lbl,
                                all_answer_type_lbl, all_span_idx, all_question_ends)

        print("*********Dataset preprocessing complete**********")
        return dataset, all_graph_out
        
    else:
        raise Exception("Wrong argument: args.task only accepts `para_select` and `train_model`")
