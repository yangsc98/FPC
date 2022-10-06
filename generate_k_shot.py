import os
import json

import numpy as np


k = 16
seed_list = [13, 21, 42]

original_data_dir = "./saved_data/tacred/"
processed_data_path = "./processed_data/tacred/"


def load_dataset(data_path):
    split_list = ["train", "dev", "test"]
    dataset = {}

    for split in split_list:
        file_name = os.path.join(data_path, "%s.json" % split)

        with open(file_name, "r") as file:
            data = json.load(file)
        dataset[split] = data

    return dataset


def main():
    dataset = load_dataset(original_data_dir)

    print("K =", k)
    for seed in seed_list:
        print("seed = %d" % seed)
        np.random.seed(seed)

        train_lines = dataset["train"]
        np.random.shuffle(train_lines)

        dev_lines = dataset["dev"]
        np.random.shuffle(dev_lines)

        output_path = os.path.join(processed_data_path, "%d-%d" % (k, seed))
        os.makedirs(output_path, exist_ok=True)

        label_dict = {}
        for line in train_lines:
            label = line["relation"]

            if label not in label_dict:
                label_dict[label] = [line]
            else:
                label_dict[label].append(line)

        train_list = []
        for label in label_dict:
            for line in label_dict[label][:k]:
                train_list.append(line)

        with open(os.path.join(output_path, "train.json"), "w") as file:
            json.dump(train_list, file)

        dev_label_dict = {}
        for line in dev_lines:
            label = line["relation"]

            if label not in dev_label_dict:
                dev_label_dict[label] = [line]
            else:
                dev_label_dict[label].append(line)

        dev_list = []
        for label in dev_label_dict:
            for line in dev_label_dict[label][:k]:
                dev_list.append(line)

        with open(os.path.join(output_path, "dev.json"), "w") as file:
            json.dump(dev_list, file)

        with open(os.path.join(output_path, "test.json"), "w") as file:
            json.dump(dataset["test"], file)


if __name__ == "__main__":
    main()
