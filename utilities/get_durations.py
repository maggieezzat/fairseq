import librosa
import argparse
import glob
import os
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root", help="root directory containing wav files"
    )
    parser.add_argument(
        "--dest", default=".", type=str, help="output file"
    )
    parser.add_argument(
        "--ext", default="wav", type=str, metavar="EXT", help="extension to look for"
    )
    return parser

def main(args):

    dir_path = os.path.realpath(args.root)
    print(dir_path)
    #search_path = os.path.join(dir_path, "**/*." + args.ext)
    search_path = Path(dir_path).rglob(args.ext)
    print(search_path)
    #################################################

    seconds = 0.0

    #for fname in glob.iglob(search_path, recursive=True):
    for fname in search_path:
        #################################################
        file_path = os.path.realpath(fname)
        dur = librosa.get_duration(filename=file_path)
        #print(dur)
        seconds += dur

    hours = int(seconds // 3600)
    rem = seconds % 3600
    minutes = int(rem // 60)
    secs = int(rem % 60)

    with open(args.dest, 'w') as out:
        out.write("Total Duration is: " + str(hours) + " hr, " + str(minutes) + " min, " + str(secs) + " seconds.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

