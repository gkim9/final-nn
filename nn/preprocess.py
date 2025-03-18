# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    pos_seqs = []
    neg_seqs = []

    for seq, label in zip(seqs, labels):
        if label:
            pos_seqs.append(seq)
        else:
            neg_seqs.append(seq)

    if len(pos_seqs) > len(neg_seqs):
        minority_seqs = neg_seqs
        minority_labels = [0] * len(neg_seqs)

        majority_seqs = pos_seqs
        majority_labels = [1] * len(pos_seqs)

    else:
        minority_seqs = pos_seqs
        minority_labels = [1] * len(neg_seqs)

        majority_seqs = neg_seqs
        majority_labels = [0] * len(pos_seqs)

    # calculate how many samples to add to balance classes
    num_samples_needed = len(majority_seqs) - len(minority_seqs)

    # randomly select samples from the minority class
    sampled_minority_idx = np.random.choice(
        len(minority_seqs), size=num_samples_needed, replace=True
    )

    # create new lists with sampled data

    sampled_minority_seqs = []
    sampled_minority_labels = []

    for i in sampled_minority_idx:
        sampled_minority_seqs.append(minority_seqs[i])
        sampled_minority_labels.append(minority_labels[0])

    # add original minority seqs
    sampled_minority_seqs += minority_seqs
    sampled_minority_labels += minority_labels

    # add majority and sampled minority
    sampled_seqs = majority_seqs + sampled_minority_seqs
    sampled_labels = majority_labels + sampled_minority_labels

    # shuffle the combined sequences and labels
    combined = list(zip(sampled_seqs, sampled_labels))
    np.random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)

    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    nuc_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    encodings = []
    for seq in seq_arr:
        encoded_seq = []

        for base in seq:
            encoded_seq.extend(nuc_map[base])

        encodings.append(encoded_seq)

    return np.array(encodings)
