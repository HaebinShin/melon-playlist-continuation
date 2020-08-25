from tqdm import tqdm
import numpy as np
from implicit.als import AlternatingLeastSquares as ALS
import lightgbm as lgb


class CollectiveMF:
    def __init__(self, num_entities, num_factors, **kwargs):
        self.num_entities = num_entities
        self.num_factors = num_factors
        self.model = ALS(factors=num_factors, **kwargs)

    def fit(self, csr_matrix, confidence, show_progress=True):
        self.model.fit(csr_matrix.T * confidence, show_progress=show_progress)
        return self.model

    def _split_collective_model(self, csr_matrix):
        splited_models = []
        splited_csr_matrix = []
        prev_idx = 0
        for num_entity in self.num_entities:
            model = ALS(factors=self.num_factors)
            model.user_factors = self.model.user_factors
            model.item_factors = self.model.item_factors[prev_idx:prev_idx+num_entity]
            splited_models.append(model)

            entity_csr_matrix = csr_matrix[:, prev_idx:prev_idx+num_entity]
            splited_csr_matrix.append(entity_csr_matrix)

            prev_idx = prev_idx + num_entity
        return splited_models, splited_csr_matrix

    def _recommend_each(self,
                        model,
                        uids,
                        lookup_csr,
                        N,
                        filter_already_liked_items,
                        filter_items,
                        recalculate_user,
                        item_idx_offset,
                        progress_description=''):
        res = []
        for playlist_idx in tqdm(uids, total=len(uids), desc=progress_description):
            rec = model.recommend(playlist_idx,
                                  lookup_csr,
                                  N=N,
                                  filter_already_liked_items=filter_already_liked_items,
                                  filter_items=filter_items,
                                  recalculate_user=recalculate_user)
            rec = [(x[0]+item_idx_offset, x[1]) for x in rec]
            res.append(rec)
        return res

    def recommend(self,
                  uids,
                  lookup_csr,
                  num_rec_group,
                  filter_already_liked_items,
                  filter_items,
                  recalculate_user,
                  progress_description='Recommend'):
        res_all_entity = []
        splited_models, splited_lookup_csr = self._split_collective_model(lookup_csr)
        for idx, (model, csr_matrix, num_rec) in enumerate(zip(splited_models, splited_lookup_csr, num_rec_group)):
            res = self._recommend_each(model=model,
                                       uids=uids,
                                       lookup_csr=csr_matrix,
                                       N=num_rec,
                                       filter_already_liked_items=filter_already_liked_items,
                                       filter_items=filter_items,
                                       recalculate_user=recalculate_user,
                                       item_idx_offset=sum(self.num_entities[0:idx]),
                                       progress_description=''.join([progress_description, 
                                                                     f" - Entity [{idx+1}/{len(splited_models)}]"]))
            res_all_entity.append(res)
        return res_all_entity

    def recommend_cold_start(self):
        raise NotImplementedError

    def save(self, path):
        np.savez_compressed(path,
                            num_entities=self.num_entities,
                            num_factors=self.num_factors,
                            user_factors=self.model.user_factors,
                            item_factors=self.model.item_factors)

    def load(self, path):
        data = np.load(path)
        self._verify_loaded_data(data['num_entities'], self.num_entities)
        self._verify_loaded_data(data['num_factors'], self.num_factors)
        self.model.user_factors = data['user_factors']
        self.model.item_factors = data['item_factors']

    def _verify_loaded_data(self, loaded, defined):
        loaded = np.array(loaded)
        defined = np.array(defined)
        assert (loaded == defined).all(), \
            f"loaded: {loaded.shape} vs defined: {defined.shape}"


class LTRBoosting():
    def __init__(self, label_gain=None):
        self.gbm = lgb.LGBMRanker(label_gain=label_gain)

    def fit(self, **kwargs):
        self.gbm.fit(**kwargs)

    def predict(self, X):
        pred = self.gbm.predict(X)
        return pred

    def save(self, fname):
        raise NotImplementedError

    def load(self, fname):
        raise NotImplementedError
