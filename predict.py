import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from data_loader import ParagraphSelectorFeatures
from para_select import ParagraphSelector
# TODO: Make prediction after `model` and `trainer` are set

logger = logging.getLogger(__name__)


class ParaPredictor(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        self.model = self.load_model()
        logger.info("***** Ranking paragraphs - Paragraph Selection *****")

    def title_matching(self, example):
        doc_idx_list = []
        titles, docs = zip(*example.context)
        doc_idx_list = [doc_idx for doc_idx, title in enumerate(titles) if title.lower() in example.question.lower()]
        return doc_idx_list

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        try:
            model = ParagraphSelector.from_pretrained(self.args.model_dir)
            model.to(self.device)
        except:
            raise Exception("Some model files might be missing...")

        return model

    def construct_para_features(self, max_seq_len, example,
                                cls_token_segment_id=0,
                                pad_token_segment_id=0,
                                sequence_a_segment_id=0,
                                sequence_b_segment_id=1,
                                mask_padding_with_zero=True):
        # Build input features for `para_predict()`
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token_id = self.tokenizer.pad_token_id

        features = []
        question = example.question
        context = example.context
        _, paragraphs = zip(*context)
        context_list = [''.join(para) for para in paragraphs]
        for i in range(len(context)):
            question_tokens = self.tokenizer.tokenize(question)
            context_tokens = self.tokenizer.tokenize(context_list[i])

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

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

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

            features.append(
                ParagraphSelectorFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=None
                ))
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset

    def para_predict(self, example):
        dataset = self.construct_para_features(self.args.max_seq_len, example)

        para_sampler = SequentialSampler(dataset)
        para_dataloader = DataLoader(dataset, sampler=para_sampler, batch_size=self.args.eval_batch_size)

        preds = []
        nb_eval_steps = 0

        self.model.eval()
        for batch in para_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': None}
                outputs = self.model(**inputs)  # (loss), logits, (hidden_states), (attentions)
                _, logits = outputs[:2]
            
            nb_eval_steps += 1

            sig_out = F.sigmoid(logits)
            # print("Sigmoid output: ", sig_out.item())
            preds.append(sig_out.item())

        return np.array(preds)
