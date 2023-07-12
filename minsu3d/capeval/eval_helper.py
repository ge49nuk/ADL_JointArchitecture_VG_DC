import os
import sys
import json
import torch
import numpy as np

import minsu3d.capeval.bleu.bleu as capblue
import minsu3d.capeval.cider.cider as capcider
import minsu3d.capeval.rouge.rouge as caprouge
import minsu3d.capeval.meteor.meteor as capmeteor


def prepare_corpus(raw_data, max_len=30):
    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)
        # key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def eval_cap(cfg, candidates_iou25, candidates_iou50, force=True, max_len=30):
    
    # corpus
    corpus_path = os.path.join(cfg.data.dataset_root_path, "corpus_val.json")
    scanrefer_path = os.path.join(cfg.data.scanrefer_path, "ScanRefer_filtered_val.json")
    if not os.path.exists(corpus_path) or force:
        print("preparing corpus_val...")
        raw_data = json.load(open(scanrefer_path))

        corpus = prepare_corpus(raw_data, max_len)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus_val...")
        with open(corpus_path) as f:
            corpus = json.load(f)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    # key = "{}|{}|{}".format(scene_id, object_id, object_name)
    candidates_iou25 = check_candidates(corpus, candidates_iou25)
    candidates_iou25 = organize_candidates(corpus, candidates_iou25)

    candidates_iou50 = check_candidates(corpus, candidates_iou50)
    candidates_iou50 = organize_candidates(corpus, candidates_iou50)

    with open("candidates_iou25.json", "w") as f:
        json.dump(candidates_iou25, f, indent=4)
    with open("candidates_iou50.json", "w") as f:
        json.dump(candidates_iou50, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu_iou25 = capblue.Bleu(4).compute_score(corpus, candidates_iou25)[0][3]
    cider_iou25 = capcider.Cider().compute_score(corpus, candidates_iou25)[0]
    rouge_iou25 = caprouge.Rouge().compute_score(corpus, candidates_iou25)[0]
    meteor_iou25 = capmeteor.Meteor().compute_score(corpus, candidates_iou25)[0]

    bleu_iou50 = capblue.Bleu(4).compute_score(corpus, candidates_iou50)[0][3]
    cider_iou50 = capcider.Cider().compute_score(corpus, candidates_iou50)[0]
    rouge_iou50 = caprouge.Rouge().compute_score(corpus, candidates_iou50)[0]
    meteor_iou50 = capmeteor.Meteor().compute_score(corpus, candidates_iou50)[0]
    
    return (bleu_iou25, bleu_iou50), (cider_iou25, cider_iou50), (rouge_iou25, rouge_iou50), (meteor_iou25, meteor_iou50)

