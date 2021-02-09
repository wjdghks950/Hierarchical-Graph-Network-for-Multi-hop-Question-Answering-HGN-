import argparse
import neptune

from para_select import ParaSelectorTrainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


'''
Finetune BERT-based (or RoBERTa-based) Paragraph Selection model.

'''
def main(args):
    init_logger()
    set_seed(args)

    if args.logger:
        neptune.init("wjdghks950/NumericHGN")
        neptune.create_experiment(name="({}) NumHGN_{}_{}_{}".format(args.task, args.train_batch_size, args.max_seq_len, args.train_file))
        neptune.append_tag("BertForSequenceClassification", "finetuning", "num_augmented_HGN")

    tokenizer = load_tokenizer(args)
    train_dataset = dev_dataset = test_dataset = None
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    # test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = ParaSelectorTrainer(args, train_dataset, dev_dataset)

    if args.do_train:
        trainer.train()
        trainer.save_model()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")

    if args.logger:
        neptune.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="para_select", type=str, help="Task: `Paragraph Selection (para_select)` or `Train Model (train_model)`")  # Can be `opspam`, `yelp` or `amazon`
    parser.add_argument("--model_dir", default="./model_checkpoint", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="hotpot_train_v1.1.json", type=str, help="Train file")
    parser.add_argument("--dev_file", default="hotpot_dev_distractor_v1.json", type=str, help="Dev file (distractor & full_wiki)")
    parser.add_argument("--test_file", default="test.csv", type=str, help="Test file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--logger', action="store_true", help="Activate neptune logger")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
