#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("trans", help="transcription file path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    transcriptions = {}
    labels = {}
    labels['|'] = 0

    with open(args.trans, 'r') as f:
        for line in f:
            line = line.strip().split(" ", 1)
            file_path = line[0]
            #file_path = file_path.split("/")[-1]
            file_trans = line[1]
            transcriptions[file_path] = file_trans
    
    
    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        root = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            wav_path = line.split()[0]
            if wav_path in transcriptions:
                texts = transcriptions[wav_path]
                print(texts, file=wrd_out)
                print(
                    " ".join(list(texts.replace(" ", "|"))) + " |",
                    file=ltr_out,
                )

                if args.output_name == 'train':
                    words = texts.split()
                    for word in words:
                        chars = list(word)
                        for char in chars:
                            if char in labels.keys():
                                labels[char] +=1
                            else:
                                labels[char] = 1
                    piped_sent = " ".join(list(texts.replace(" ", "|"))) + " |"
                    for ch in piped_sent:
                        if ch == '|':
                            labels['|'] +=1

    if args.output_name == 'train':
        labels = dict( sorted(labels.items(), key=lambda item: item[1], reverse=True))
        with open(os.path.join(args.output_dir, "dict.ltr.txt"), "w") as out:
            for k,v in labels.items():
                out.write(k + " " + str(v) + '\n')


if __name__ == "__main__":
    main()
