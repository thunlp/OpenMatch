import numpy as np
import argparse
import pickle
from tqdm import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_embedding", type=str)
    parser.add_argument("--output_embeddings", type=str)
    parser.add_argument("--num_splits", type=int, default=2)
    args = parser.parse_args()

    with open(args.input_embedding, "rb") as f:
        embedding, lookup = pickle.load(f)
        lookup = np.array(lookup)
    
    for split in trange(args.num_splits):
        embedding_split = embedding[split::args.num_splits]
        lookup_split = lookup[split::args.num_splits]
        with open(args.output_embeddings + f".{split}", "wb") as f:
            pickle.dump((embedding_split, lookup_split.tolist()), f, protocol=4)