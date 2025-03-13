# Source Acknowledgment
# This material includes code modified from the open-source project https://github.com/karpathy/llama2.c.
# The Tokenizer class is used under the Llama 2 Community License Agreement, available at https://ai.meta.com/llama/license/.

import os
from typing import List
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import torch
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, tokenizer_model):
        assert os.path.isfile(tokenizer_model), tokenizer_model
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_model)
        self.model_path = tokenizer_model

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


def _process_shard(args, tokenizer_model, output_bin_dir):
    '''
    pretokenize each shard of text data using tokenizer_model
    '''
    shard_id, shard = args
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text using BOS
        all_tokens.extend(tokens)

    # Use np.uint16 may limit the vocabulary if the num of tokens exceeds 65535. Consider a larger dtype like np.int32 or np.int64
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    shard_basename = os.path.basename(shard)
    bin_basename = shard_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(output_bin_dir, bin_basename)
    
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    # Calculate the average sequence length (as separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
    

def pretokenize(data_dir, dataset_name, tokenizer_model):
    # iterate the shards and tokenize them
    dataset_dir = os.path.join(data_dir, dataset_name)
    shard_filenames = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    output_bin_dir = os.path.join(data_dir, dataset_name+"_pretok")
    os.makedirs(output_bin_dir, exist_ok=True)

    # Process all the shards in a process pool
    fun = partial(_process_shard, tokenizer_model=tokenizer_model, output_bin_dir=output_bin_dir)

    # Change the number of workers based on your system's capability
    num_workers = 2
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(fun, enumerate(shard_filenames))

    bin_files = glob.glob(os.path.join(output_bin_dir, "*.bin"))
    print(f"Expected to create {len(shard_filenames)} .bin files, actually created {len(bin_files)} .bin files.")
    if len(bin_files) < len(shard_filenames):
        print("Some .bin files were not created. Check logs for errors.")
    print(f"Pretokenized data are saved in .bin files under {output_bin_dir}")
    return output_bin_dir


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk (pretok_bin_dir) and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, pretok_bin_dir):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.bin_dir = pretok_bin_dir

    def __iter__(self):
        shard_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*.bin")))

        # train/test split. Use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {self.bin_dir}"

        # The loop allows the dataset to provide an infinite stream, useful for training that require multiple epochs.
        while True:
            random.shuffle(shard_filenames)
            for shard in shard_filenames:
                # Uses np.memmap to map the file into memory, allowing efficient reading of large datasets without loading the entire file into RAM
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                random.shuffle(ixs)
                for ix in ixs:
                    # Slices a chunk of data from the memmapped array.
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # Converts it to a PyTorch tensor.
                    # Calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    # Each pair of sequences represents a single example
                    yield x, y


class BatchProcessor:
    '''Higher-level data preparation, including batching and device move'''
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y
