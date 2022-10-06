from config import roberta_large_path, saved_output_path

import os
import time
import argparse

import torch

from data.Data_tac import Dataset_tac
from data.Data_semeval import Dataset_semeval
from model.Model_roberta import robertaMLMRE
from train.Train_base import train_base
from train.Train_mlm import train_mlm
from train.Train_mlmkg import train_mlmkg
from train.Train_mlmkg_2 import train_mlmkg_2
from train.Evaluate import evaluate
from utils.Tools import set_seed


dataset_dict = {
    "tacred": Dataset_tac,
    "tacrev": Dataset_tac,
    "retacred": Dataset_tac,
    "semeval": Dataset_semeval,
}

train_dict = {
    "base": train_base,
    "mlm": train_mlm,
    "mlmkg": train_mlmkg,
    "mlmkg_2": train_mlmkg_2,
}


def main():
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--evaluate", action='store_true')

    # Dataset
    parser.add_argument('--dataset_name', type=str, default="tacred", help='["tacred", "tacrev", "retacred", "semeval"]')
    parser.add_argument('--template', type=str, default="Simple", help='["Simple", "Ent", "Typ", "EntTyp"]')
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Model
    parser.add_argument("--folder_name", type=str, default="exp_1")
    parser.add_argument('--downloaded_model', action='store_true')
    parser.add_argument("--use_gradient_checkpoint", action='store_true')

    parser.add_argument("--model_train", type=str, default="base", help='["base", "mlm", "mlmkg", "mlmkg_2"]')
    parser.add_argument("--mlm_ratio", type=float, default=0.4)
    parser.add_argument("--num_guide_epochs", type=int, default=5)

    # Dataloader
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # Train
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_step_ratio", type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Device
    parser.add_argument('--device', type=str, default="cuda", help='["cpu", "cuda"]')

    # Other settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_model", action='store_true')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    start_time = time.localtime()
    time_str = time.strftime("%m(M)%d(D)-%H:%M", start_time)
    print("Experiment Running Time: %s" % time_str)
    print("Experiment Arguments:\n%s" % args)

    if not args.downloaded_model:
        model_name_or_path = "roberta-large"
    else:
        model_name_or_path = roberta_large_path
    args.model_name_or_path = model_name_or_path

    saved_folder_path = os.path.join(saved_output_path, args.folder_name)
    os.makedirs(saved_folder_path, exist_ok=True)

    args.saved_result_path = os.path.join(saved_folder_path, "result.json")
    args.saved_model_path = os.path.join(saved_folder_path, "roberta_large.pth")
    args.saved_args_path = os.path.join(saved_folder_path, "training_args.json")

    if args.seed is not None:
        set_seed(args.seed)

    dataset_class = dataset_dict[args.dataset_name]
    train_func = train_dict[args.model_train]

    train_dataset = dataset_class(args, "train")
    args.label_num = train_dataset.label_num
    dev_dataset = dataset_class(args, "dev")
    test_dataset = dataset_class(args, "test")

    model = robertaMLMRE(args)
    if args.train:
        train_func(args, model, train_dataset, dev_dataset, test_dataset)
    elif os.path.exists(args.saved_model_path):
        model.load_state_dict(torch.load(args.saved_model_path))
    else:
        raise ValueError("can't find the state dict of the model!")

    if args.evaluate:
        dev_precision, dev_recall, dev_micro_f1, dev_pred_array = evaluate(args, model, dev_dataset)
        print("dev final result: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (dev_precision, dev_recall, dev_micro_f1))

        test_precision, test_recall, test_micro_f1, test_pred_array = evaluate(args, model, test_dataset)
        print("test final result: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (test_precision, test_recall, test_micro_f1))


if __name__ == "__main__":
    main()
