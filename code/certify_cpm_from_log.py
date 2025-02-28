# evaluate a smoothed classifier on a dataset from log files
import argparse
import os
import json
from datasets import get_num_classes
from code.core_cpm import Smooth
from time import time
import torch
import datetime
import numpy as np
from architectures import get_architecture
import shutil
from tqdm import tqdm

os.environ["IMAGENET_DIR"] = "imagenet/"


def load_counts(file_path):
    """
    Function to load counts from the N0 or N file
    """
    counts_selection = []
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            counts_str = parts[2]  # This is the 'counts' part
            counts = eval(counts_str)  # Convert string list to an actual list of ints
            counts_selection.append(np.array(counts))  # Store as numpy array
    return np.array(counts_selection)


def load_labels(file_path):
    """
    Function to load counts from the N0 or N file
    """
    counts_selection = []
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            counts_str = parts[1]  # This is the 'counts' part
            counts = eval(counts_str)  # Convert string list to an actual list of ints
            counts_selection.append(np.array(counts))  # Store as numpy array
    return np.array(counts_selection)


def load_config_from_json(config_file):
    """
    Function to load sigma, dataset, base_classifier, skip, max, N, and N0 from JSON config file
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    dataset = config.get("dataset")  # Dataset is loaded from the config file
    base_classifier = config.get("base_classifier")  # Base classifier path from config
    sigma = config.get("sigma")  # Sigma is loaded from the config file
    skip = config.get("skip", 1)  # Default value of 1 if not provided in JSON
    max_examples = config.get("max", -1)  # Default value of -1 if not provided in JSON
    N = config.get("N", 10000)  # Default value of 10000 if not provided in JSON
    N0 = config.get("N0", 100)  # Default value of 100 if not provided in JSON
    return dataset, base_classifier, sigma, skip, max_examples, N, N0


def create_output_directory(dataset, sigma, config_file, arch):
    """
    Function to create output directory and return file paths
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = os.path.join(
        "certify_from_log", dataset, "sigma_" + str(sigma), timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "output_file_with_certif.txt")

    # add mode to the config file
    shutil.copy(config_file, os.path.join(output_dir, "certify_args.json"))
    name_config = os.path.join(output_dir, "certify_args.json")
    return output_file, name_config


parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument(
    "--indir", type=str, help="input directory containing N0 and N files"
)
parser.add_argument(
    "--mode",
    type=str,
    help="radius choice",
    choices=["Rmono", "Rmulti"],
    default="Rmono",
)
parser.add_argument(
    "--certif",
    type=str,
    help="certification procedure used",
    choices=["bonferroni", "pearson_clopper", "cpm"],
    default="pearson_clopper",
)
parser.add_argument(
    "--alpha", type=float, default=0.001, help="significance level alpha"
)
args = parser.parse_args()

if __name__ == "__main__":
    config_file = os.path.join(args.indir, "certify_args.json")
    dataset, base_classifier_path, sigma, skip, max_examples, N, N0 = (
        load_config_from_json(config_file)
    )

    checkpoint = torch.load(base_classifier_path)

    if "liresnet" in base_classifier_path:
        arch = "liresnet"
        state_dict = checkpoint["backbone"]
    else:
        arch = checkpoint["arch"]
        state_dict = checkpoint["state_dict"]

    print("dataset, base_classifier_path, sigma, skip, max_examples, N, N0")
    print(
        dataset,
        base_classifier_path,
        sigma,
        skip,
        max_examples,
        N,
        N0,
        args.certif,
        args.alpha,
    )
    base_classifier = get_architecture(arch, dataset)
    base_classifier.load_state_dict(state_dict)

    N0_file = [
        f for f in os.listdir(args.indir) if f.startswith("certify") and "N0" in f
    ][0]
    N_file = [
        f for f in os.listdir(args.indir) if f.startswith("certify") and "_N_" in f
    ][0]

    smoothed_classifier = Smooth(base_classifier, get_num_classes(dataset), sigma)

    output_file, name_config = create_output_directory(
        dataset, sigma, config_file, arch
    )

    with open(name_config, "r") as f:
        config = json.load(f)
    config["mode"] = args.mode
    config["certif"] = args.certif
    with open(name_config, "w") as f:
        json.dump(config, f, indent=4)

    counts_selection = load_counts(os.path.join(args.indir, N0_file))
    counts_estimation = load_counts(os.path.join(args.indir, N_file))
    labels = load_labels(os.path.join(args.indir, N0_file))

    with open(output_file, "w") as f:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

        if dataset == "imagenet":
            size = 50000
        else:
            # cifar10
            size = 10000

        for i in tqdm(range(size)):
            if i % skip != 0:
                continue
            if i == max_examples:
                break

            idx_counts = i // skip

            label = labels[idx_counts]

            before_time = time()

            if args.certif == "bonferroni":
                prediction, radius = smoothed_classifier.certify_from_log_bonferroni(
                    counts_selection[idx_counts],
                    counts_estimation[idx_counts],
                    N,
                    N0,
                    args.alpha,
                    mode=args.mode,
                )
            elif args.certif == "cpm":
                prediction, radius = smoothed_classifier.certify_from_log_cpm(
                    counts_selection[idx_counts],
                    counts_estimation[idx_counts],
                    N,
                    N0,
                    args.alpha,
                    mode=args.mode,
                )
            elif args.certif == "pearson_clopper":
                prediction, radius = smoothed_classifier.certify_from_log(
                    counts_selection[idx_counts],
                    counts_estimation[idx_counts],
                    N,
                    args.alpha,
                    mode=args.mode,
                )

            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print(
                "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                    i, label, prediction, radius, correct, time_elapsed
                ),
                file=f,
                flush=True,
            )
