#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import moxing as mox
mox.file.shift('os', 'mox')

import os
import argparse
import collections
import sys

data_dir = 's3://bi-gru/scripts/thumt/data'
#print(os.listdir(data_dir))

def _open(filename, mode="r", encoding="utf-8"):
    filename = os.path.join(data_dir, filename)

    if sys.version_info.major == 2:
        return open(filename, mode=mode)
    elif sys.version_info.major == 3:
        return open(filename, mode=mode, encoding=encoding)
    else:
        raise RuntimeError("Unknown Python version for running!")


def count_words(filename):
    counter = collections.Counter()

    with _open(filename, "r") as fd:
        for line in fd.readlines():
            words = line.strip().split()
            counter.update(words)
        fd.close()
        
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))
    
    return words, counts


def control_symbols(string):
    if not string:
        return []
    else:
        return string.strip().split(",")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, ids = list(zip(*pairs))

    with _open(name, "w") as f:
        for word in words:
            f.write(word + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")

    parser.add_argument("corpus", help="input corpus")
    parser.add_argument("output", default="vocab.txt",
                        help="Output vocabulary name")
    parser.add_argument("--limit", default=0, type=int, help="Vocabulary size")
    parser.add_argument("--control", type=str, default="<pad>,<eos>,<unk>",
                        help="Add control symbols to vocabulary. "
                             "Control symbols are separated by comma.")

    return parser.parse_args()


def main(args):
    vocab = {}
    limit = args.limit
    count = 0
    
    words, counts = count_words(args.corpus)
    
    #ctrl_symbols = control_symbols(args.control)
    #for sym in ctrl_symbols:
    #    vocab[sym] = len(vocab)
    
    for i, (word, freq) in enumerate(zip(words, counts)):
        if i % 1000: print('processing', i)
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq
        
    print("saving into the vocab file---")
    save_vocab(args.output, vocab)

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    main(parse_args())
