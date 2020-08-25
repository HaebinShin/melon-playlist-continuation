import os
import json
import argparse
import pandas as pd
import numpy as np

from src.feature import PlaylistFeature, SongLTRDataset, TagLTRDataset
from src.model import CollectiveMF, LTRBoosting


def read_data(dir, additional=[]):
    part1_path = os.path.join(dir, 'inputs', 'part1.json')
    part1_df = pd.read_json(part1_path)

    part2_q_path = os.path.join(dir, 'inputs', 'part2_q.json')
    part2_q_df = pd.read_json(part2_q_path)

    part3_q_path = os.path.join(dir, 'inputs', 'part3_q.json')
    part3_q_df = pd.read_json(part3_q_path)

    evaluation_q_path = os.path.join(dir, 'inputs', 'evaluation_q.json')
    evaluation_q_df = pd.read_json(evaluation_q_path)

    train_df = pd.concat([part1_df, part2_q_df, part3_q_df, evaluation_q_df])

    part2_a_path = os.path.join(dir, 'labels', 'part2_a.json')
    part2_a_df = pd.read_json(part2_a_path)

    part3_a_path = os.path.join(dir, 'labels', 'part3_a.json')
    part3_a_df = pd.read_json(part3_a_path)

    evaluation_a_path = os.path.join(dir, 'labels', 'evaluation_a.json')
    evaluation_a_df = pd.read_json(evaluation_a_path)

    if len(additional) > 0:
        for file_ in additional:
            df = pd.read_json(file_)
            train_df = pd.concat([train_df, df])

    result = (train_df,
              (evaluation_q_df, evaluation_a_df),
              (part2_q_df, part3_q_df, part2_a_df, part3_a_df))
    return result


def get_candidates(cmf, df, pf, desc):
    pids = df.id.apply(lambda x: pf.playlistid_to_idx[x]).tolist()
    rec_song, rec_tag = cmf.recommend(uids=pids,
                                      lookup_csr=pf.csr_matrix,
                                      num_rec_group=[500, 100],
                                      filter_already_liked_items=True,
                                      filter_items=None,
                                      recalculate_user=False,
                                      progress_description=desc)
    rec_song = [[(pf.idx_to_songid[x[0]], x[1]) for x in rec_list]
                for rec_list in rec_song]
    rec_tag = [[(pf.idx_to_tag[x[0]], x[1]) for x in rec_list]
               for rec_list in rec_tag]
    return rec_song, rec_tag


def run_boosting(df_train, group_train,
                 df_valid, group_valid,
                 df_test):
    model = LTRBoosting(label_gain=[i for i in range(df_train.target.max()+1)])

    model.fit(X=df_train.drop(columns=['pid', 'item', 'target']),
              y=df_train.target,
              group=group_train,
              eval_set=[(df_valid.drop(columns=['pid', 'item', 'target']),
                         df_valid.target)],
              eval_group=[group_valid],
              eval_at=[100],
              early_stopping_rounds=200,
              verbose=True)
    pred = model.predict(df_test.drop(columns=['pid', 'item', 'target']))
    return pred


def dump_output(fname, id_list, item_ret, tag_ret):
    returnval = [
        {
            "id": _id,
            "songs": rec[:100],
            "tags": tag_rec[:10]
        }
        for _id, rec, tag_rec in zip(id_list, item_ret, tag_ret)
    ]

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(json.dumps(returnval, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', required=True)
    parser.add_argument('--additional', dest='additional',
                        nargs="*", default=[])
    args = parser.parse_args()
    print(args.additional)

    train_df, evaluation, part23 = read_data(args.dir, args.additional)
    part2_q_df, part3_q_df, part2_a_df, part3_a_df = part23
    evaluation_q_df, evaluation_a_df = evaluation

    pf = PlaylistFeature(train_df)

    print("------Phase 1. Get Candidates------")
    print([pf.num_song, pf.num_tag])
    cmf = CollectiveMF(num_entities=[pf.num_song, pf.num_tag],
                       num_factors=800,
                       regularization=0.01,
                       calculate_training_loss=True,
                       iterations=30)
    cmf.fit(csr_matrix=pf.csr_matrix, confidence=15.)
    cmf.save('cmf.npz')
    # # cmf.load('./cmf.npz')

    part2_rec_song, part2_rec_tag = get_candidates(cmf,
                                                   part2_q_df,
                                                   pf,
                                                   "Part2 Recommend")

    np.savez_compressed('./part2_rec',
                        part2_rec_song=part2_rec_song,
                        part2_rec_tag=part2_rec_tag)
    dump_output('./cmf_part2.json',
                part2_q_df.id.tolist(),
                [[tup[0] for tup in item_rec] for item_rec in part2_rec_song],
                [[tup[0] for tup in tag_rec] for tag_rec in part2_rec_tag])

    part3_rec_song, part3_rec_tag = get_candidates(cmf,
                                                   part3_q_df,
                                                   pf,
                                                   "Part3 Recommend")

    dump_output('./cmf_part3.json',
                part3_q_df.id.tolist(),
                [[tup[0] for tup in item_rec] for item_rec in part3_rec_song],
                [[tup[0] for tup in tag_rec] for tag_rec in part3_rec_tag])
    np.savez_compressed('./part3_rec',
                        part3_rec_song=part3_rec_song,
                        part3_rec_tag=part3_rec_tag)

    evaluation_rec_song, evaluation_rec_tag = get_candidates(cmf,
                                                             evaluation_q_df,
                                                             pf,
                                                             "Eval Recommend")

    dump_output('./cmf_evaluation.json',
                evaluation_q_df.id.tolist(),
                [[tup[0] for tup in item_rec] for item_rec in evaluation_rec_song],
                [[tup[0] for tup in tag_rec] for tag_rec in evaluation_rec_tag])
    np.savez_compressed('./evaluation_rec',
                        evaluation_rec_song=evaluation_rec_song,
                        evaluation_rec_tag=evaluation_rec_tag)

    # part2_rec = np.load('./part2_rec.npz')
    # part2_rec_song = part2_rec['part2_rec_song']
    # part2_rec_tag = part2_rec['part2_rec_tag']

    # part3_rec = np.load('./part3_rec.npz')
    # part3_rec_song = part3_rec['part3_rec_song']
    # part3_rec_tag = part3_rec['part3_rec_tag']

    # evaluation_rec = np.load('./evaluation_rec.npz')
    # evaluation_rec_song = evaluation_rec['evaluation_rec_song']
    # evaluation_rec_tag = evaluation_rec['evaluation_rec_tag']

    print("------Phase 2. Learning To Rank------")
    print("------Phase 2.1. Song Model------")
    print("------Extract Song Feature - Train------")
    rank = part2_a_df.songs.apply(lambda list_: {
                                    song: rank+1
                                    for rank, song in enumerate(list_)
                                  }).tolist()
    sltrd_train = SongLTRDataset(pids=part2_q_df.id.tolist(),
                                 item_ret=part2_rec_song,
                                 item_gt=rank,
                                 pf=pf,
                                 input_df=part2_q_df)
    song_ltr_df_train, song_ltr_group_train = sltrd_train.df, sltrd_train.group

    print("------Extract Song Feature - Valid------")
    rank = part3_a_df.songs.apply(lambda list_: {
                                    song: rank+1
                                    for rank, song in enumerate(list_)
                                  }).tolist()
    sltrd_valid = SongLTRDataset(pids=part3_q_df.id.tolist(),
                                 item_ret=part3_rec_song,
                                 item_gt=rank,
                                 pf=pf,
                                 input_df=part3_q_df)
    song_ltr_df_valid, song_ltr_group_valid = sltrd_valid.df, sltrd_valid.group

    print("------Extract Song Feature - Test------")
    sltrd_test = SongLTRDataset(pids=evaluation_q_df.id.tolist(),
                                item_ret=evaluation_rec_song,
                                pf=pf,
                                input_df=evaluation_q_df)
    song_ltr_df_test, _ = sltrd_test.df, sltrd_test.group

    print("------Boosting Song Model------")
    song_pred = run_boosting(song_ltr_df_train, song_ltr_group_train,
                             song_ltr_df_valid, song_ltr_group_valid,
                             song_ltr_df_test)
    song_ltr_df_test['predicted_ranking'] = song_pred
    song_pred_df = song_ltr_df_test.sort_values('predicted_ranking',
                                                ascending=False)
    song_pred_df = song_pred_df.groupby('pid', sort=False)
    song_pred_df = song_pred_df.item.apply(lambda x: x.values[:100])
    song_pred_df = song_pred_df.reset_index()

    print("------Phase 2.2. Tag Model------")
    print("------Extract Tag Feature - Train------")
    rank = part2_a_df.tags.apply(lambda list_: {
                                    tag: rank+1
                                    for rank, tag in enumerate(list_)
                                 }).tolist()
    tltrd_train = TagLTRDataset(pids=part2_q_df.id.tolist(),
                                item_ret=part2_rec_tag,
                                item_gt=rank,
                                pf=pf,
                                input_df=part2_q_df)
    tag_ltr_df_train, tag_ltr_group_train = tltrd_train.df, tltrd_train.group

    print("------Extract Tag Feature - Valid------")
    rank = part3_a_df.tags.apply(lambda list_: {
                                    tag: rank+1
                                    for rank, tag in enumerate(list_)
                                 }).tolist()
    tltrd_valid = TagLTRDataset(pids=part3_q_df.id.tolist(),
                                item_ret=part3_rec_tag,
                                item_gt=rank,
                                pf=pf,
                                input_df=part3_q_df)
    tag_ltr_df_valid, tag_ltr_group_valid = tltrd_valid.df, tltrd_valid.group

    print("------Extract Tag Feature - Test------")
    tltrd_test = TagLTRDataset(pids=evaluation_q_df.id.tolist(),
                               item_ret=evaluation_rec_tag,
                               pf=pf,
                               input_df=evaluation_q_df)
    tag_ltr_df_test, _ = tltrd_test.df, tltrd_test.group

    print("------Boosting Tag Model------")
    tag_pred = run_boosting(tag_ltr_df_train, tag_ltr_group_train,
                            tag_ltr_df_valid, tag_ltr_group_valid,
                            tag_ltr_df_test)
    tag_ltr_df_test['predicted_ranking'] = tag_pred
    tag_pred_df = tag_ltr_df_test.sort_values('predicted_ranking',
                                              ascending=False)
    tag_pred_df = tag_pred_df.groupby('pid', sort=False)
    tag_pred_df = tag_pred_df.item.apply(lambda x: x.values[:10])
    tag_pred_df = tag_pred_df.reset_index()

    print("------Phase 2.3. Dump Final Output------")
    pred_df = pd.merge(song_pred_df, tag_pred_df,
                       on='pid', suffixes=('_song', '_tag'))

    res_pids = pred_df.pid.tolist()
    res_s = pred_df.item_song.apply(lambda x: x.astype(int).tolist()).tolist()
    res_t = pred_df.item_tag.apply(lambda x: x.tolist()).tolist()
    dump_output('result.json', res_pids, res_s, res_t)
