# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
import torch
import datetime
from architectures import get_architecture
import json
from tqdm import tqdm
import numpy as np

os.environ["IMAGENET_DIR"] = "/home/lamsade/bdelattre/ssd/imagenet/"

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset", default="cifar10")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter", default=0.25)
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists("logs"):
        os.makedirs("logs")

    iid= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    name_dir = os.path.join("logs", args.dataset)
    name_dir_run = os.path.join(name_dir, iid)

    if not os.path.exists(name_dir_run):
        os.makedirs(name_dir_run)
    print(f"Saving logs to {name_dir_run}")

    args_dict = vars(args)
    with open(os.path.join(name_dir_run, "certify_args.json"), 'w') as f:
        json.dump(args_dict, f)

    args.outfile = os.path.join(name_dir_run, f"certify_{args.dataset}_{args.sigma}_N_{args.N}_alpha_{args.alpha}.txt")
    args.outfile_N0 = os.path.join(name_dir_run, f"certify_{args.dataset}_{args.sigma}_N0_{args.N0}_alpha_{args.alpha}.txt")

    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    log_f = open(args.outfile, 'w')
    log_f.write("idx\tlabel\tcounts\n")

    log_f_N0 = open(args.outfile_N0, 'w')
    log_f_N0.write("idx\tlabel\tcounts\n")

    dataset = get_dataset(args.dataset, args.split)

    print("len dataset / skip", len(dataset) / args.skip)
    for i in tqdm(range(len(dataset))):

        if i % args.skip != 0:
            continue
        if i == args.max:   
            break

        (x, label) = dataset[i]

        x = x.cuda()
        counts_selection_N0 = smoothed_classifier.log_certify(x, args.N0, args.batch)
        counts_selection_N = smoothed_classifier.log_certify(x, args.N, args.batch)

        print(f"{i}\t{label}\t{np.argmax(counts_selection_N0.tolist())}")

        log_f_N0.write(f"{i}\t{label}\t{counts_selection_N0.tolist()}\n")
        log_f.write(f"{i}\t{label}\t{counts_selection_N.tolist()}\n")
        
        if i % 100 == 0:
            log_f.flush()
            log_f_N0.flush()



    log_f.close()
    log_f_N0.close()
