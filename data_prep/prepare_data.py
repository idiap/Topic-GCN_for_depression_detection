# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Builds the interview-level text files used by every experiment from the original
DAIC-WOZ or E-DAIC distribution.

Each output line is one interview, as `<label>\t<text>`, where the label is
`positive` (depressed) or `negative`, and the text is the concatenation of the
transcript turns:

  - DAIC-WOZ: only the turns whose speaker is `Participant` (manual transcripts);
  - E-DAIC:   every turn of the `<pid>_Transcript.csv` file (ASR transcripts).

Labels follow the paper (`--label-rule auto`): DAIC-WOZ uses the official
`PHQ8_Binary` column and E-DAIC uses PHQ-8 score >= 10. The two rules disagree
for a few participants of each corpus (e.g. DAIC-WOZ participant 409, or the
E-DAIC participants whose `PHQ_Binary` is 0 with a score of 10), so pass
`--label-rule binary` or `--label-rule score` to apply one rule to both corpora
(this no longer reproduces the paper).

Interviews are written in the order of the official split files, which is the
order the models were trained and evaluated with in the paper.

Usage:
    python data_prep/prepare_data.py --corpus daic-woz --root /path/to/DAIC-WOZ --out data/daic-woz
    python data_prep/prepare_data.py --corpus e-daic   --root /path/to/E-DAIC   --out data/e-daic
"""
import os
import io
import glob
import tarfile
import argparse
import pandas as pd

SPLITS = ("train", "dev", "test")

# Labelling rule used in the paper for each corpus
LABEL_RULE = {"daic-woz": "binary", "e-daic": "score"}

# Split files of each corpus, as distributed by USC ICT.
SPLIT_FILES = {
    "daic-woz": {
        "train": ["train_split_Depression_AVEC2017.csv"],
        "dev": ["dev_split_Depression_AVEC2017.csv"],
        "test": ["full_test_split.csv"],
    },
    "e-daic": {
        "train": ["labels/train_split.csv", "train_split.csv"],
        "dev": ["labels/dev_split.csv", "dev_split.csv"],
        "test": ["labels/test_split.csv", "test_split.csv"],
    },
}


def find_file(root, candidates):
    for candidate in candidates:
        path = os.path.join(root, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"None of {candidates} found under '{root}'")


def read_split(path):
    """Returns a list of (participant_id, phq_binary, phq_score), in file order."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    binary_col = next(c for c in df.columns if c in ("phq8_binary", "phq_binary"))
    score_col = next(c for c in df.columns if c in ("phq8_score", "phq_score"))
    return [(int(row["participant_id"]), int(row[binary_col]), int(row[score_col]))
            for _, row in df.iterrows()]


def read_transcript_daic(root, pid):
    path = os.path.join(root, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.isfile(path):
        path = find_file(root, [f"data/{pid}_P/{pid}_TRANSCRIPT.csv", f"{pid}_TRANSCRIPT.csv"])
    df = pd.read_csv(path, sep="\t")
    return [str(row["value"]) for _, row in df.iterrows() if row["speaker"] == "Participant"]


def read_transcript_edaic(root, pid):
    name = f"{pid}_Transcript.csv"
    candidates = [os.path.join(root, f"{pid}_P", name),
                  os.path.join(root, "data", f"{pid}_P", name),
                  os.path.join(root, name)]
    content = None
    for path in candidates:
        if os.path.isfile(path):
            content = path
            break
    if content is None:
        # E-DAIC is distributed as one <pid>_P.tar archive per participant
        tars = glob.glob(os.path.join(root, f"{pid}_P.tar")) + glob.glob(os.path.join(root, "*", f"{pid}_P.tar"))
        if not tars:
            raise FileNotFoundError(f"No transcript found for participant {pid} under '{root}'")
        with tarfile.open(tars[0]) as tar:
            content = io.BytesIO(tar.extractfile(f"{pid}_P/{name}").read())
    df = pd.read_csv(content, sep=",")
    return [str(row["Text"]) for _, row in df.iterrows()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=SPLIT_FILES.keys(), required=True)
    parser.add_argument("--root", required=True, help="Root folder of the corpus as downloaded from USC ICT")
    parser.add_argument("--out", required=True, help="Output folder (e.g. data/daic-woz)")
    parser.add_argument("--label-rule", choices=["auto", "binary", "score"], default="auto",
                        help="'binary': the corpus' PHQ binary column; 'score': PHQ-8 score >= 10; "
                             "'auto' (default): as in the paper, 'binary' for DAIC-WOZ and 'score' for E-DAIC")
    args = parser.parse_args()
    label_rule = args.label_rule if args.label_rule != "auto" else LABEL_RULE[args.corpus]

    read_transcript = read_transcript_daic if args.corpus == "daic-woz" else read_transcript_edaic
    os.makedirs(args.out, exist_ok=True)

    for split in SPLITS:
        participants = read_split(find_file(args.root, SPLIT_FILES[args.corpus][split]))
        labels, texts = [], []
        for pid, phq_binary, phq_score in participants:
            depressed = phq_binary == 1 if label_rule == "binary" else phq_score >= 10
            labels.append("positive" if depressed else "negative")
            texts.append("".join(turn + " " for turn in read_transcript(args.root, pid)))

        pd.DataFrame({"label": labels, "text": texts}).to_csv(
            os.path.join(args.out, f"{split}.tsv"), sep="\t", header=False, index=False)
        pd.DataFrame(participants, columns=["participant_id", "phq8_binary", "phq8_score"]).to_csv(
            os.path.join(args.out, f"{split}_ids.tsv"), sep="\t", index=False)

        print(f"{split:5s}: {len(texts):3d} interviews "
              f"({labels.count('positive')} positive, {labels.count('negative')} negative) "
              f"-> {os.path.join(args.out, split + '.tsv')}")


if __name__ == "__main__":
    main()
