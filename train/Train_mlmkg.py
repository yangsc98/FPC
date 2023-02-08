import time
import json
from copy import deepcopy
import random

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from train.Evaluate import evaluate


def train_mlmkg(args, model, train_dataset, dev_dataset, test_dataset):
    model.to(args.device)
    model.zero_grad()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    train_batch_size = args.train_batch_size
    num_train_epochs = args.num_train_epochs
    num_guide_epochs = args.num_guide_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps

    total_train_batch_size = train_batch_size * gradient_accumulation_steps
    num_training_steps = int(len(train_dataloader) // gradient_accumulation_steps * num_train_epochs)
    num_warmup_steps = int(num_training_steps * args.warmup_step_ratio)

    scaler = GradScaler()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    print("***** Running Training *****")
    print("  dataset instance num = %d" % len(train_dataset))
    print("  total train batch size = %d" % total_train_batch_size)
    print("  gradient accumulation steps = %d" % gradient_accumulation_steps)
    print("  train epoch num = %d" % num_train_epochs)
    print("  guide epoch num = %d" % num_guide_epochs)
    print("  training step num = %d" % num_training_steps)
    print("  warmup step num = %d" % num_warmup_steps)

    dev_best_result = 0.0
    test_final_result = 0.0
    dev_pred_list = []
    test_pred_list = []

    global_step = 0
    training_loss = 0.0
    logging_loss = 0.0

    train_iterator = range(1, num_train_epochs + 1)
    for epoch in train_iterator:
        start_time = time.time()

        mlm_ratio = args.mlm_ratio
        MLM_input_prob = epoch / num_guide_epochs
        epoch_iterator = iter(train_dataloader)
        for step, inputs in enumerate(epoch_iterator):
            model.train()

            input_ids = inputs["input_ids"].to(args.device)
            MLM_input_ids = inputs["MLM_input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            subj_start = inputs["subj_start"].to(args.device)
            obj_start = inputs["obj_start"].to(args.device)
            MLM_labels = inputs["MLM_labels"].to(args.device)
            cls_label = inputs["cls_label"].to(args.device)

            if random.random() < MLM_input_prob:
                outputs = model(
                    input_ids=MLM_input_ids,
                    attention_mask=attention_mask,
                    subj_start=subj_start,
                    obj_start=obj_start,
                    MLM_labels=MLM_labels,
                    cls_label=cls_label,
                    mlm_ratio=mlm_ratio,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    subj_start=subj_start,
                    obj_start=obj_start,
                    MLM_labels=MLM_labels,
                    cls_label=cls_label,
                    mlm_ratio=mlm_ratio,
                )

            loss = outputs["loss"]
            if args.gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            training_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(epoch_iterator):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    loss_scalar = (training_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    print("  training steps: %d / %d, loss = %.6f, learning rate = %.6f" % (global_step, num_training_steps, loss_scalar, learning_rate_scalar))
                    logging_loss = training_loss

                if global_step % args.eval_steps == 0:
                    dev_precision, dev_recall, dev_micro_f1, dev_pred_array = evaluate(args, model, dev_dataset)
                    print("dev during train: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (dev_precision, dev_recall, dev_micro_f1))

                    test_precision, test_recall, test_micro_f1, test_pred_array = evaluate(args, model, test_dataset)
                    print("test during train: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (test_precision, test_recall, test_micro_f1))

                    if dev_micro_f1 > dev_best_result:
                        dev_best_result = dev_micro_f1
                        test_final_result = test_micro_f1
                        dev_pred_list = dev_pred_array.tolist()
                        test_pred_list = test_pred_array.tolist()
                        saved_state_dict = deepcopy(model.state_dict())

        end_time = time.time()
        epoch_time = end_time - start_time
        print("  train epoch %d / %d, global step = %d, epoch time = %.2fs" % (epoch, num_train_epochs, global_step, epoch_time))

    dev_precision, dev_recall, dev_micro_f1, dev_pred_array = evaluate(args, model, dev_dataset)
    print("dev final result: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (dev_precision, dev_recall, dev_micro_f1))

    test_precision, test_recall, test_micro_f1, test_pred_array = evaluate(args, model, test_dataset)
    print("test final result: precision = %.6f, recall = %.6f, micro f1 = %.6f" % (test_precision, test_recall, test_micro_f1))

    if dev_micro_f1 > dev_best_result:
        dev_best_result = dev_micro_f1
        test_final_result = test_micro_f1
        dev_pred_list = dev_pred_array.tolist()
        test_pred_list = test_pred_array.tolist()
        saved_state_dict = deepcopy(model.state_dict())

    print("%s result: dev best result = %.6f, test final result = %.6f" % (args.dataset_name, dev_best_result, test_final_result))

    saved_result_dict = {
        "dev_best_result": dev_best_result,
        "test_final_result": test_final_result,
        "dev_pred_list": dev_pred_list,
        "test_pred_list": test_pred_list
    }
    with open(args.saved_result_path, "w", encoding="utf-8") as file:
        json.dump(saved_result_dict, file)

    if args.save_model:
        torch.save(saved_state_dict, args.saved_model_path)

        with open(args.saved_args_path, "w", encoding="utf8") as file:
            json.dump(args.__dict__, file)
