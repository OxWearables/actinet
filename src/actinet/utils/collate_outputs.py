import argparse
import os
import json
from collections import OrderedDict
import pandas as pd
from tqdm.auto import tqdm


def collate_outputs(outputs, outfile="outputs.csv"):
    """Read all *-outputSummary.json files under <outputs> and merge into one CSV file.
    :param str outputs: Directory containing JSON files.
    :param str outfile: Output CSV filename.
    :return: New file written to <outfile>
    :rtype: void
    """

    # Load all *-outputSummary.json files under outputs/
    infofiles = []
    jdicts = []
    for root, _, files in os.walk(outputs):
        for file in files:
            if file.endswith("-outputSummary.json"):
                infofiles.append(os.path.join(root, file))

    print(f"Found {len(infofiles)} summary files...")
    for file in tqdm(infofiles):
        with open(file, "r") as f:
            jdicts.append(json.load(f, object_pairs_hook=OrderedDict))

    summary = pd.DataFrame.from_dict(jdicts)  # merge to a dataframe
    summary.to_csv(outfile, index=False)
    print("Summary CSV written to", outfile)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", help="Directory containing JSON files.")
    parser.add_argument("--outfile", "-o", default="outputs.csv",
                        help="Output CSV filename.")
    args = parser.parse_args()

    return collate_outputs(outputs=args.outputs, outfile=args.outfile)


if __name__ == "__main__":
    main()
