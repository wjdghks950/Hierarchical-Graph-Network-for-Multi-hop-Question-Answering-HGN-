import torch
import torch.nn

import os
import logging
from tqdm import tqdm, trange
import neptune

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel, BertModel

from utils import compute_metrics, MODEL_CLASSES, f1_score, EarlyStopping

logger = logging.getLogger(__name__)


class ParagraphSelector(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = 0.0
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.type_as(logits)
            loss = loss_fct(logits.squeeze(-1), labels)
        outputs = (loss,) + (logits,) + outputs
        return outputs  # (loss,), (binary_logits), logits_bert, (hidden_states), (attentions)


class ParaSelectorTrainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.early_stopping = EarlyStopping(patience=10, verbose=True)

        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type]
        # self.config = self.config_class.from_pretrained(self.args.model_name_or_path, num_labels=1, output_hidden_states=True, output_attentions=True)

        self.model = ParagraphSelector.from_pretrained(self.args.model_name_or_path, num_labels=1)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        print("***************** Config & Pretrained Model load complete **********************")

    def train(self):
        print("Entering Trainer...")
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        acc = 0.0
        global_step = 0
        tr_loss = 0.0
        tr_acc = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}

                outputs = self.model(**inputs)
                loss, logits = outputs[:2]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                    pred = 1 if logits.squeeze(-1).item() > 0 else 0
                    if pred == batch[3].item():
                        acc = 1.0
                        acc /= self.args.gradient_accumulation_steps
                    else:
                        acc = 0.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                tr_loss += loss.item()
                tr_acc += acc
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logger:
                        # logger.info('Loss: %f', tr_loss / global_step)
                        neptune.log_metric("Loss", tr_loss / global_step)
                        neptune.log_metric("(Train) Accuracy", tr_acc / global_step)

                # if self.args.logging_steps > 0 and (step + 1) % self.args.logging_steps == 0 and self.dev_dataset is not None:
                #     self.evaluate("dev") # TODO: Problem: dev file save is saved with train data!!

                # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev dataset available for evaluation in HotpotQA")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}

                outputs = self.model(**inputs)  # (loss), logits, (hidden_states), (attentions)

                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            pred = 1 if logits.squeeze(-1).item() > 0 else 0
            pred = np.array([pred])
            if preds is None:
                preds = pred
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, pred, axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        prec = precision(preds, out_label_ids)
        rec = recall(preds, out_label_ids)
        f1 = f1_score(preds, out_label_ids)

        # if self.early_stopping.validate((results['loss'])):
        #     print("Early stopping... Terminating Process.")
        #     exit(0)

        if self.args.logger:
            neptune.log_metric('(Val.) Loss', results['loss'])
            neptune.log_metric('(Val.) Accuracy', results['acc'])
            neptune.log_metric('(Val.) F1 Score', f1)
            neptune.log_metric('(Val.) Precision', prec)
            neptune.log_metric('(Val.) Recall', rec)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("  prec = %s", str(prec))
        logger.info("  rec = %s", str(rec))
        logger.info("  f1 = %s", str(f1))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        try:
            self.model = self.model.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
