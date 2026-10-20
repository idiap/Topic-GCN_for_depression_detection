<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: Apache-2.0
-->

# 📁 Data preparation

The experiments use two corpora of clinical interviews from USC ICT, which we cannot redistribute:

| Corpus | Interviews (train / dev / test) | Transcripts | Request access |
|--------|------|------|------|
| DAIC-WOZ | 189 (107 / 35 / 47) | manual, speaker-labelled | [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/) |
| E-DAIC | 275 (163 / 56 / 56) | ASR (`<pid>_Transcript.csv`) | [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/) |

Once you have them, [`prepare_data.py`](prepare_data.py) builds the files every experiment reads:

```bash
python data_prep/prepare_data.py --corpus daic-woz --root /path/to/DAIC-WOZ --out data/daic-woz
python data_prep/prepare_data.py --corpus e-daic   --root /path/to/E-DAIC   --out data/e-daic
```

## 📥 Expected input layout

**DAIC-WOZ** (`--root`):

```
<root>/
├── 300_P/300_TRANSCRIPT.csv          # one folder per participant (also accepted under <root>/data/)
├── ...
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
└── full_test_split.csv               # test labels, distributed with the corpus
```

**E-DAIC** (`--root`):

```
<root>/
├── 300_P/300_Transcript.csv          # or the original 300_P.tar archives
├── ...
└── labels/
    ├── train_split.csv
    ├── dev_split.csv
    └── test_split.csv
```

The transcript is read straight from the `<pid>_P.tar` archives if they have not been extracted.

## 📤 Output

One folder per corpus with three files, one interview per line, as `<label>\t<text>`:

```
data/daic-woz/train.tsv   data/daic-woz/dev.tsv   data/daic-woz/test.tsv
data/e-daic/train.tsv     data/e-daic/dev.tsv     data/e-daic/test.tsv
```

and, next to each, `<split>_ids.tsv` with the participant IDs and PHQ-8 scores in the same order.

- **Text.** DAIC-WOZ keeps only the turns of the participant (speaker `Participant`), so the interviewer's prompts are left out. E-DAIC transcripts have no speaker labels, so all their turns are kept. Turns are joined with a space, without any other normalisation.
- **Labels.** `positive` means depressed. As in the paper, DAIC-WOZ uses the official `PHQ8_Binary` column and E-DAIC uses PHQ-8 score ≥ 10. The two rules disagree for a few participants: participant 409 in DAIC-WOZ, and 13 train, 3 dev and 4 test participants in E-DAIC whose `PHQ_Binary` does not match their score. `--label-rule binary|score` applies a single rule to both corpora, but then the output no longer matches the paper.
- **Order.** Interviews follow the order of the official split files, as in the paper.

> [!NOTE]
> We checked that the files written by this script are byte-for-byte identical to the ones behind the paper, for the three splits of both corpora.
> For E-DAIC, the training and development interviews of the paper come from the AVEC 2019 release of the corpus, whose participants are named `training_XXX` and `development_XX`. We ran this script on the public `<pid>_Transcript.csv` layout above, mapped to those AVEC names, and the output matched.

| File | Interviews | Positive | Negative |
|------|-----------:|---------:|---------:|
| `daic-woz/train.tsv` | 107 | 30 | 77 |
| `daic-woz/dev.tsv` | 35 | 12 | 23 |
| `daic-woz/test.tsv` | 47 | 14 | 33 |
| `e-daic/train.tsv` | 163 | 50 | 113 |
| `e-daic/dev.tsv` | 56 | 15 | 41 |
| `e-daic/test.tsv` | 56 | 21 | 35 |

> [!CAUTION]
> The prepared files contain the interview transcripts and fall under the corpus licence (EULA).
> Keep them out of version control (`data/` is in `.gitignore`) and do not share them.
