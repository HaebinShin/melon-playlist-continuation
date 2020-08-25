# -*- coding: utf-8 -*-
# Made by reference to
# https://github.com/kakao-arena/melon-playlist-continuation/blob/master/split_data.py.
import io
import os
import json
import copy
import random
import distutils.dir_util
import fire
import numpy as np


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./preprocessed/" + parent)
    with io.open("./preprocessed/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


class DataSplitter:
    def _split_data(self, playlists):
        tot = len(playlists)
        train = playlists[:int(tot*0.80)]
        val = playlists[int(tot*0.80):]

        return train, val

    def _mask(self, playlists, mask_cols, del_cols):
        q_pl = copy.deepcopy(playlists)
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            for del_col in del_cols:
                q_pl[i][del_col] = []
                if del_col == 'songs':
                    a_pl[i][del_col] = a_pl[i][del_col][:100]
                elif del_col == 'tags':
                    a_pl[i][del_col] = a_pl[i][del_col][:10]

            for col in mask_cols:
                mask_len = len(playlists[i][col])
                mask = np.full(mask_len, False)
                mask[:mask_len//2] = True
                np.random.shuffle(mask)

                q_pl[i][col] = list(np.array(q_pl[i][col])[mask])
                a_pl[i][col] = list(np.array(a_pl[i][col])[np.invert(mask)])

        return q_pl, a_pl

    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists[:int(tot * 0.3)]
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print(f"Total: {len(playlists)}, "
              f"Song only: {len(song_only)}, "
              f"Song & Tags: {len(song_and_tags)}, "
              f"Tags only: {len(tags_only)}, "
              f"Title only: {len(title_only)}")

        song_q, song_a = self._mask(song_only, ['songs'], ['tags'])
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], [])
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs'])
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags'])

        q = song_q + songtag_q + tag_q + title_q
        a = song_a + songtag_a + tag_a + title_a

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])
        a = list(np.array(a)[shuffle_indices])

        return q, a

    def run(self, fname):
        random.seed(777)
        np.random.seed(777)

        print("Reading data...")
        playlists = load_json(fname)
        random.shuffle(playlists)
        print(f"Total playlists: {len(playlists)}\n")

        print("Splitting data...")
        _part123, _evaluation = self._split_data(playlists)
        part1, _part23 = self._split_data(_part123)
        _part2 = _part23[:len(_part23)//2]
        _part3 = _part23[len(_part23)//2:]
        print(f"Part1 playlists: {len(part1)}")
        print(f"Part2 playlists: {len(_part2)}")
        print(f"Part3 playlists: {len(_part3)}")
        print(f"Evaluation playlists: {len(_evaluation)}\n")

        print("Masked...")
        part2_q, part2_a = self._mask_data(_part2)
        part3_q, part3_a = self._mask_data(_part3)
        evaluation_q, evaluation_a = self._mask_data(_evaluation)

        print("Saved...")
        write_json(part1, "inputs/part1.json")
        write_json(part2_q, "inputs/part2_q.json")
        write_json(part3_q, "inputs/part3_q.json")
        write_json(evaluation_q, "inputs/evaluation_q.json")
        write_json(part2_a, "labels/part2_a.json")
        write_json(part3_a, "labels/part3_a.json")
        write_json(evaluation_a, "labels/evaluation_a.json")


if __name__ == "__main__":
    fire.Fire(DataSplitter())
