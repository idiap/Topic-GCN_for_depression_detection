# SPDX-FileCopyrightText: Copyright © 2023-2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Trains and evaluates the GCN and Topic-GCN systems of Table 2.

Systems:
    gcn                 GCN baseline: one-hot words, TF-IDF documents, no topic nodes   (ohe, tf-idf, -)
    topic-gcn-ft        FastText words and documents, no topic nodes                      (ft, ft, -)
    topic-gcn-lsa       FastText words, documents and LSA topic nodes                     (ft, ft, ft)
    topic-gcn-lda       FastText words, documents and LDA topic nodes                     (ft, ft, ft)
    topic-gcn-bertopic  FastText words, documents and BERTopic topic nodes                (ft, ft, ft)

Hyperparameters (learning rate and number of steps):
    default             the values of the Table 2 runs (conf/hparams.yaml)
    --lr X --steps N    given values
    --hpo               Optuna search, 100 trials (as in the paper)

Examples:
    python run_topic_gcn.py --dataset daic-woz --system topic-gcn-bertopic
    python run_topic_gcn.py --dataset e-daic --system topic-gcn-lda --hpo --eval-split dev
"""
import os
import json
import yaml
import argparse
import torch

from topic_gcn.data import DATASETS, dataset_path, load_dataset
from topic_gcn.topics import load_topic_model
from topic_gcn.training import train_topic_gcn

SYSTEMS = {  # system: (uses FastText, topic model)
    "gcn": (False, "none"),
    "topic-gcn-ft": (True, "none"),
    "topic-gcn-lsa": (True, "lsa"),
    "topic-gcn-lda": (True, "lda"),
    "topic-gcn-bertopic": (True, "bertopic"),
}
HP_RANGE_LR = (1e-7, 1e-3)
HP_RANGE_STEPS = (50, 13000)
PATH_HPARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "hparams.yaml")


def model_paths(models_root, dataset):
    """Default model locations, as written by train_fasttext.py and train_topic_model.py."""
    root = os.path.join(models_root, dataset)
    return {
        "fasttext": os.path.join(root, "fasttext", "fasttext.model"),
        "lsa": os.path.join(root, "lsa", "lsa_topics.pkl"),
        "lda": os.path.join(root, "lda", "lda.gensim"),
        "bertopic": os.path.join(root, "bertopic"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--system", choices=SYSTEMS.keys(), required=True)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="test",
                        help="Split used to select the best training step (and to optimise the hyperparameters "
                             "with --hpo). The paper reports runs with 'test' (default)")
    parser.add_argument("--lr", type=float, help="Learning rate (default: conf/hparams.yaml)")
    parser.add_argument("--steps", type=int, help="Training steps (default: conf/hparams.yaml)")
    parser.add_argument("--hpo", action="store_true", help="Search learning rate and steps with Optuna")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials (with --hpo)")
    parser.add_argument("--seed", type=int, help="Random seed (default: conf/hparams.yaml)")
    parser.add_argument("--n-features", type=int, default=250, help="Word nodes: top-k words by ANOVA F-value")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--data-root", default="data", help="Output folder of data_prep/prepare_data.py")
    parser.add_argument("--models-root", default="models", help="Output folder of the FastText and topic models")
    parser.add_argument("--fasttext", help="FastText model (default: <models-root>/<dataset>/fasttext/fasttext.model)")
    parser.add_argument("--topic-model", help="Topic model file or folder (default: under <models-root>/<dataset>/)")
    parser.add_argument("--topic-dict", help="gensim dictionary of the LDA model (default: next to it)")
    parser.add_argument("--out", default="exp", help="Results go to <out>/<dataset>/<system>/<eval-split>/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    use_fasttext, topic_model_name = SYSTEMS[args.system]
    with open(PATH_HPARAMS) as reader:
        hparams = yaml.safe_load(reader)[args.dataset][args.system]
    seed = args.seed if args.seed is not None else hparams["seed"]
    out_dir = os.path.join(args.out, args.dataset, args.system, args.eval_split)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    X_train, y_train, ix2label, label2ix = load_dataset(dataset_path(args.data_root, args.dataset, "train"))
    X_eval, y_eval, _, _ = load_dataset(dataset_path(args.data_root, args.dataset, args.eval_split), label2ix)
    labels = [ix2label[ix] for ix in range(len(ix2label))]

    # Node initialisation and topic model
    paths = model_paths(args.models_root, args.dataset)
    word_vectors, topic_model = None, None
    if use_fasttext:
        from gensim.models import FastText
        word_vectors = FastText.load(args.fasttext or paths["fasttext"]).wv
    if topic_model_name != "none":
        topic_path = args.topic_model or paths[topic_model_name]
        topic_dict = args.topic_dict or os.path.join(os.path.dirname(topic_path), "dictionary.gensim")
        topic_model = load_topic_model(topic_model_name, topic_path, topic_dict)

    def run(lr, steps):
        return train_topic_gcn(X_train, y_train, X_eval, y_eval, labels, lr, steps,
                               word_vectors=word_vectors, topic_model=topic_model,
                               n_features=args.n_features, hidden_dim=args.hidden_dim,
                               seed=seed, device=args.device)

    if args.hpo:
        import optuna

        def objective(trial):
            lr = trial.suggest_float("learning_rate", *HP_RANGE_LR, log=True)
            steps = trial.suggest_int("num_steps", *HP_RANGE_STEPS, log=True)
            return run(lr, steps)[0]

        study = optuna.create_study(storage=f"sqlite:///{os.path.join(out_dir, 'optuna.sqlite3')}",
                                    study_name=f"{args.dataset}-{args.system}-{args.eval_split}",
                                    sampler=optuna.samplers.TPESampler(seed=seed),
                                    direction="maximize", load_if_exists=True)
        study.optimize(objective, n_trials=max(0, args.n_trials - len(study.trials)))
        lr, steps = study.best_params["learning_rate"], study.best_params["num_steps"]
        print(f"Best trial: lr={lr}, steps={steps} (macro F1 {study.best_value:.4f}); retraining it...")
    else:
        lr = args.lr if args.lr is not None else hparams["lr"]
        steps = args.steps if args.steps is not None else hparams["steps"]

    best_value, report, model = run(lr, steps)

    results = {"dataset": args.dataset, "system": args.system, "eval_split": args.eval_split,
               "lr": lr, "steps": steps, "seed": seed, "macro_f1": best_value, "report": report}
    with open(os.path.join(out_dir, "results.json"), "w") as writer:
        json.dump(results, writer, indent=2)

    nodes = model.node_predictions()
    nodes["labels"] = labels
    with open(os.path.join(out_dir, "node_predictions.json"), "w") as writer:
        json.dump(nodes, writer)

    print(f"\n{args.dataset} | {args.system} | eval on {args.eval_split} | lr={lr:.3g} steps={steps} seed={seed}")
    print(f"  macro F1 = {best_value:.4f}   " +
          "   ".join(f"F1({label}) = {report[label]['f1-score']:.4f}" for label in labels) +
          f"   (best step {report['step']})")
    print(f"  results: {out_dir}/results.json")


if __name__ == "__main__":
    main()
