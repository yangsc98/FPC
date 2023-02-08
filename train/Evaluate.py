import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.Metric import f1_score


def evaluate(args, model, dataset):
    model.to(args.device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )

    label_array = np.array([], dtype=np.int64)
    pred_array = np.array([], dtype=np.int64)

    epoch_iterator = iter(dataloader)
    for step, inputs in enumerate(epoch_iterator):
        label_array = np.append(label_array, inputs["cls_label"].numpy())

        MLM_input_ids = inputs["MLM_input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        subj_start = inputs["subj_start"].to(args.device)
        obj_start = inputs["obj_start"].to(args.device)

        with torch.no_grad():
            outputs = model(
                input_ids=MLM_input_ids,
                attention_mask=attention_mask,
                subj_start=subj_start,
                obj_start=obj_start,
                MLM_labels=None,
                cls_label=None,
                mlm_ratio=None,
            )
            cls_logits = outputs["cls_logits"]

        preds = torch.argmax(cls_logits, dim=-1)
        pred_array = np.append(pred_array, preds.to("cpu").numpy())

    precision, recall, micro_f1 = f1_score(pred_array, label_array, args.label_num)

    return precision, recall, micro_f1, pred_array
