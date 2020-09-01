# Melon Playlist Continuation
This is an extra solution to the [Melon Playlist Continuation Challenge](https://arena.kakao.com/c/7) by the [***](https://github.com/ssstttaaarrr) Team.  
It was inspired by the following two papers: [A hybrid two-stage recommender system for automatic playlist continuation](https://dl.acm.org/doi/pdf/10.1145/3267471.3267488), which won 3rd place in the [RecSys Challenge ’18](http://www.recsyschallenge.com/2018/); and [Relational Learning via Collective Matrix Factorization](http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf).  

## Dataset
As stated in the [Challenge README](https://arena.kakao.com/c/7/data), the dataset in `data.tar.gz` contains 150K playlists that have been created by [Melon](https://www.melon.com/) users.  
To untar the dataset:
```bash
tar -xvzf data.tar.gz
```
The `data/train.json` contains all the data, whereas `data/val.json` and `data/test.json` are just for submission, so only some of the songs and tags are included.  
For this repository, we just consider `data/val.json` and `data/test.json` as additional information.

## Solution
- Phase 1: Extract candidates using CMF Recommandation(song+tag matrix)
- Phase 2: Re-rank candidates using Learning-To-Rank Boosting

## Preprocessing - Data Partitioning
For local evaluation, we create the new `evaluation` dataset. The `part2` and `part3` are for the training and validation datasets for boosting, respectively.  
These are divided into question (_q) and answer (_a) parts.  
In Phase 1, we train `part1`+`part2_q`+`part3_q`+`evaluation_q` and optionally include `valid.json`+`test.json` as additional information.  
In Phase 2, we use `part2_q` and `part3_q` as inputs and use `part2_a` and `part3_a` as labels, respectively.  
Please refer to [A hybrid two-stage recommender system for automatic playlist continuation](https://dl.acm.org/doi/pdf/10.1145/3267471.3267488) for detailed partitioning.

![](./docs/partitioning.svg)

## Usage
### Preprocessing
```bash
python3 preprocess.py run ./data/train.json
```
After running the above, the preprocessed directory is as follows.  
```
├── preprocessed
    ├── inputs
       ├── part1.json
       ├── part2_q.json
       ├── part3_q.json
       └── evaluation_q.json
    └── labels
       ├── part2_a.json
       ├── part3_a.json
       └── evaluation_a.json
```

### Training and Prediction
```bash
python3 run.py --dir ./preprocessed --additional ./data/val.json ./data/test.json
```
The `--additional` flag is optional.
```bash
python3 run.py --dir ./preprocessed
```

### Evaluation
```bash
python3 evaluate.py --result ./result.json --answer ./preprocessed/labels/evaluation_a.json
```

## Score
```
Music nDCG: 0.250488
Tag nDCG: 0.413651
Final Score: 0.274963
```
Final Score = Music nDCG * 0.85 + Tag nDCG * 0.15

## Running Environment
We tested this implementation using Python 3.6.9 with an Intel Core i7-9700 CPU and 32GB RAM.
