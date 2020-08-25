import warnings
import numpy as np
import pandas as pd
import scipy.sparse as spr
from tqdm import tqdm


class PlaylistFeature:
    def __init__(self, df):
        self.df = df

        self.playlistid_to_idx = self._build_data_to_idx([self.df.id.tolist()], offset=0)
        self.idx_to_playlistid = {v: k for k, v in self.playlistid_to_idx.items()}

        self.songid_to_idx = self._build_data_to_idx(self.df.songs.tolist(), offset=0)
        self.idx_to_songid = {v: k for k, v in self.songid_to_idx.items()}
        self.num_song = len(self.songid_to_idx)

        self.tag_to_idx = self._build_data_to_idx(self.df.tags.tolist(), offset=len(self.songid_to_idx))
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
        self.num_tag = len(self.tag_to_idx)

        _playlist_item_lists = self._make_playlist_item_lists(self.df, self.songid_to_idx, self.tag_to_idx)
        self.num_playlist = len(_playlist_item_lists)
        _shape = (self.num_playlist, self.num_song+self.num_tag)
        self.csr_matrix = self._lil_to_csr_matrix(_playlist_item_lists, _shape)

        self.co_occurrence_matrix = self.csr_matrix.T * self.csr_matrix
        self.co_occurrence_matrix.setdiag(0)
        self.occurrence = np.array(self.co_occurrence_matrix.sum(axis=0, dtype=np.int32).tolist()[0], dtype=np.int32)

    def _build_data_to_idx(self, lil, offset=0):
        data_to_idx = {}
        for list_ in lil:
            for item_id in list_:
                if item_id not in data_to_idx:
                    data_to_idx[item_id] = len(data_to_idx) + offset
        return data_to_idx

    def _make_playlist_item_lists(self, df, songid_to_idx, tag_to_idx):
        """
        p1 | song1, song2, ..., tag1, tag2 ...
        p2 | song2, song5, ..., tag3, tag9 ...
        p3 | ...
        ...
        """
        matrix = []
        all_playlists_songs = df.songs.tolist()
        all_playlists_tags = df.tags.tolist()
        for playlist_songs, playlist_tags in zip(all_playlists_songs, all_playlists_tags):
            row = [songid_to_idx[sid] for sid in playlist_songs if sid in songid_to_idx]
            row += [tag_to_idx[tag] for tag in playlist_tags if tag in tag_to_idx]
            matrix.append(row)
        return matrix

    def _lil_to_csr_matrix(self, playlist_item_lists, shape):
        row = []
        col = []
        dat = []
        for idx, li in enumerate(playlist_item_lists):
            for item in li:
                row.append(idx)
                col.append(item)
                dat.append(1)
        return spr.csr_matrix((dat, (row, col)), dtype=np.int8, shape=shape)

    def get_co_occurrence_feature(self, item_ret, item_seed, rec_to_idx, seed_to_idx, progress_description=''):
        co_occurrence_mean_feature = []
        co_occurrence_max_feature = []
        co_occurrence_median_feature = []
        for item_rec_list, seed in tqdm(zip(item_ret, item_seed), total=len(item_ret), desc=progress_description):
            rec_ids = np.array([rec_to_idx[item_id] for item_id, score in item_rec_list])
            seed = [seed_to_idx[item_id] for item_id in seed if item_id in seed_to_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                filtered_co_occurrence = self.co_occurrence_matrix[rec_ids[:, np.newaxis], seed].toarray()
                filtered_seed = self.occurrence[seed]

                norm_filtered_co_occurrence = np.nan_to_num(filtered_co_occurrence / filtered_seed)
                co_occurrence_mean_feature += np.nan_to_num(norm_filtered_co_occurrence.mean(axis=1)).tolist()
                co_occurrence_max_feature += np.nan_to_num(norm_filtered_co_occurrence.max(axis=1, initial=0)).tolist()
                co_occurrence_median_feature += np.nan_to_num(np.median(norm_filtered_co_occurrence, axis=1)).tolist()

        np.nan_to_num(co_occurrence_mean_feature, copy=False)
        np.nan_to_num(co_occurrence_max_feature, copy=False)
        np.nan_to_num(co_occurrence_median_feature, copy=False)
        return co_occurrence_mean_feature, co_occurrence_max_feature, co_occurrence_median_feature

    def get_naive_idf(self, item_ret, rec_to_idx, progress_description=''):
        csc_matrix = self.csr_matrix.tocsc()
        features = []
        num_doc = len(item_ret)
        for item_rec_list in tqdm(item_ret, desc=progress_description):
            rec_ids = np.array([rec_to_idx[item_id] for item_id, score in item_rec_list])
            num_contain = np.array(csc_matrix[:, rec_ids].sum(axis=0).tolist()[0])
            idf = np.log(num_doc / num_contain).tolist()
            features += idf
        return np.array(features) / np.linalg.norm(features)

    def get_count(self, item_ret, item_seed_count_list, rec_to_idx, progress_description=''):
        item_freq = []
        seed_count = []
        for item_rec_list, item_seed_count in tqdm(zip(item_ret, item_seed_count_list), total=len(item_ret), desc=progress_description):
            rec_ids = np.array([rec_to_idx[item_id] for item_id, score in item_rec_list])

            item_freq += self.occurrence[rec_ids].tolist()
            seed_count += [item_seed_count for _ in item_rec_list]

        item_freq_norm = np.array(item_freq) / np.linalg.norm(item_freq)
        seed_count_norm = np.array(seed_count) / np.linalg.norm(seed_count)
        return item_freq_norm, seed_count_norm


class LTRDataset:
    def __init__(self,
                 pids,
                 item_ret,
                 item_gt=None):
        self.group = [len(item_rec) for item_rec in item_ret]

        rank_list = [{'<default>': 0} for _ in range(len(item_ret))]
        if item_gt is not None:
            rank_list = [{item: rank+1 for rank, item in enumerate(item_list)} for item_list in item_gt]
        myseries = [
            {
                "pid": _id,
                "item": item_id,
                "target": 100-item_to_rank.get(item_id, 100),
                "rec_rank": rec_rank+1,
                "score": float(score)
            }
            for idx, (_id, item_rec, item_to_rank) in enumerate(zip(pids, item_ret, rank_list))
            for rec_rank, (item_id, score) in enumerate(item_rec)
        ]

        self.df = pd.DataFrame(myseries)
        self.df['pid'] = self.df['pid'].astype('int32')
        self.df['target'] = self.df['target'].astype('int8')
        self.df['rec_rank'] = self.df['rec_rank'].astype('int16')


class SongLTRDataset(LTRDataset):
    def __init__(self,
                 pids,
                 item_ret,
                 pf,
                 input_df,
                 item_gt=None):
        super().__init__(pids, item_ret, item_gt)

        self.df['co_ss_mean'], self.df['co_ss_max'], self.df['co_ss_median'] \
            = pf.get_co_occurrence_feature(item_ret=item_ret,
                                           item_seed=input_df.songs.tolist(),
                                           rec_to_idx=pf.songid_to_idx,
                                           seed_to_idx=pf.songid_to_idx,
                                           progress_description="Extract p(song|song) Feature")

        self.df['co_st_mean'], self.df['co_st_max'], self.df['co_st_median'] \
            = pf.get_co_occurrence_feature(item_ret=item_ret,
                                           item_seed=input_df.tags.tolist(),
                                           rec_to_idx=pf.songid_to_idx,
                                           seed_to_idx=pf.tag_to_idx,
                                           progress_description="Extract p(song|tag) Feature")

        self.df['idf'] = pf.get_naive_idf(item_ret=item_ret,
                                          rec_to_idx=pf.songid_to_idx,
                                          progress_description="Extract IDF Feature")

        self.df['freq'], self.df['seed_count'] \
            = pf.get_count(item_ret=item_ret,
                           item_seed_count_list=input_df.songs.apply(lambda x: len(x)),
                           rec_to_idx=pf.songid_to_idx,
                           progress_description="Extract Freq Feature")


class TagLTRDataset(LTRDataset):
    def __init__(self,
                 pids,
                 item_ret,
                 pf,
                 input_df,
                 item_gt=None):
        super().__init__(pids, item_ret, item_gt)

        self.df['co_ts_mean'], self.df['co_ts_max'], self.df['co_ts_median'] \
            = pf.get_co_occurrence_feature(item_ret=item_ret,
                                           item_seed=input_df.songs.tolist(),
                                           rec_to_idx=pf.tag_to_idx,
                                           seed_to_idx=pf.songid_to_idx,
                                           progress_description="Extract p(tag|song) Feature")

        self.df['idf'] = pf.get_naive_idf(item_ret=item_ret,
                                          rec_to_idx=pf.tag_to_idx,
                                          progress_description="Extract IDF Feature")

        self.df['freq'], self.df['seed_count'] \
            = pf.get_count(item_ret=item_ret,
                           item_seed_count_list=input_df.tags.apply(lambda x: len(x)),
                           rec_to_idx=pf.tag_to_idx,
                           progress_description="Extract Freq Feature")
