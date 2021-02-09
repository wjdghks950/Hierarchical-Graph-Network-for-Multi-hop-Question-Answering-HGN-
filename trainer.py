import os
import logging
from tqdm import tqdm, trange
import neptune

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from model import NumericHGN
from utils import compute_metrics, get_label, MODEL_CLASSES, recall, precision, f1_score, EarlyStopping

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, graph_out, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.graph_out = graph_out
        # self.early_stopping = EarlyStopping(patience=10, verbose=True)

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)
        self.hidden_states_list = None
        self.config_class, _, _ = MODEL_CLASSES[args.model_type]

        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=args.task, output_hidden_states=True, output_attentions=True)
        self.model = NumericHGN(self.args, config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        print("*****************Config & Pretrained Model load complete**********************")

    def train(self):
        print("Entering trainer...")
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

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                labels = (batch[3], batch[4], batch[5], batch[6])
                question_ends = batch[7]
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': labels,
                          'graph_out': self.graph_out[step],
                          'question_ends': question_ends}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)

                loss = outputs[0]  # TODO: Multiple losses for each loss term

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0 and self.dev_dataset is not None:
                    #     if self.args.logger:  # Plot training loss every 100 step
                    #         neptune.log_metric('Loss', tr_loss / step)
                    #     self.evaluate("dev")

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

                    logger.info('Loss: %f', tr_loss / global_step)


                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

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
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)  # (loss), logits, (hidden_states), (attentions)
                tmp_eval_loss, logits = outputs[:2]
                bert_hidden_states = outputs[2][0]
                bert_cls_rep = bert_hidden_states[:, :1, :].squeeze(-2).clone()

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            if self.hidden_states_list is None:
                self.hidden_states_list = bert_cls_rep
                self.labels_list = inputs['labels']
            else:
                self.hidden_states_list = torch.cat((self.hidden_states_list, bert_cls_rep), 0)
                self.labels_list = torch.cat((self.labels_list, inputs['labels']), 0)
                # print("self.hidden_states_list (shape) >> ", self.hidden_states_list.shape)
                # print("self.labels_list (shape) >> ", self.labels_list.shape)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
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

#         if self.hidden_states_list is not None:
#             torch.save(self.hidden_states_list, os.path.join(self.args.model_dir, "last_layer.pt"))
#             torch.save(self.labels_list, os.path.join(self.args.model_dir, "last_layer_label.pt"))
#             self.hidden_states_list = None
#             self.labels_list = None
#         else:
#             raise Exception("Error: self.hidden_states_list should NOT be None")

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
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
