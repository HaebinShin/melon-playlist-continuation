import json
import argparse
import numpy as np


class CustomEvaluator:
    def _load_json(self, fname):
        with open(fname, encoding="utf-8") as f:
            json_obj = json.load(f)

        return json_obj

    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt[:100])]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = self._load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = self._load_json(rec_fname)
        
        music_ndcg = 0.0
        tag_ndcg = 0.0
        
        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])
        
        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
        print(f"Music nDCG: {music_ndcg:.6}")
        print(f"Tag nDCG: {tag_ndcg:.6}")
        print(f"Final Score: {score:.6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', dest='result', required=True)
    parser.add_argument('--answer', dest='answer', required=True)
    args = parser.parse_args()

    evaluator = CustomEvaluator()
    evaluator.evaluate(args.answer, args.result)
