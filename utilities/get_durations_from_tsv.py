import librosa
import argparse
import glob
import os

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tsvfile", help="tsv file containing list of wav file paths"
    )
    return parser