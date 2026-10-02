# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/18
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import os
import json

from collections import defaultdict
from time import perf_counter
from loguru import logger
from nltk import ngrams

from crslab.evaluator.base import BaseEvaluator
from crslab.evaluator.utils import nice_report
from .metrics import *


class StandardEvaluator(BaseEvaluator):
    """The evaluator for all kind of model(recommender, conversation, policy)
    
    Args:
        rec_metrics: the metrics to evaluate recommender model, including hit@K, ndcg@K and mrr@K
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training
    """

    def __init__(self, language, file_path=None):
        super(StandardEvaluator, self).__init__()
        self.file_path = file_path
        self.result_data = []
        # rec
        self.rec_metrics = Metrics()
        # gen
        self.dist_set = defaultdict(set)
        # Distinct-n (Li et al. 2016) is (# distinct n-grams) / (# total n-grams
        # generated), a ratio in (0, 1]. `dist_ngram_cnt` accumulates the
        # denominator per n; `dist_set` (numerator) was previously divided by
        # the number of *responses* instead, which has no upper bound (grows
        # with corpus size, e.g. it showed dist@4 > 2 in practice) and isn't
        # Distinct-n at all.
        self.dist_ngram_cnt = defaultdict(int)
        self.gen_metrics = Metrics()
        # optim
        self.optim_metrics = Metrics()

    def rec_evaluate(self, ranks, label):
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"recall@{k}", RECMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"ndcg@{k}", NDCGMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"mrr@{k}", MRRMetric.compute(ranks, label, k))

    def rec_evaluate_tail(self, ranks, label):
        """Same as rec_evaluate, restricted to samples whose target item is in
        the long tail by train frequency (CPC-Hypergraph v2 spec 6.1)."""
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"tail_recall@{k}", RECMetric.compute(ranks, label, k))

    def gen_evaluate(self, hyp, refs, seq=None):
        if hyp:
            self.gen_metrics.add("f1", F1Metric.compute(hyp, refs))

            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
                grams = list(ngrams(seq, k)) if seq else []
                self.dist_set[f"dist@{k}"].update(grams)
                self.dist_ngram_cnt[f"dist@{k}"] += len(grams)

    def report(self, epoch=-1, mode='test'):
        for k, v in self.dist_set.items():
            total = self.dist_ngram_cnt[k]
            self.gen_metrics.add(k, AverageMetric(len(v), total if total > 0 else 1))
        reports = [self.rec_metrics.report(), self.gen_metrics.report(), self.optim_metrics.report()]
        all_reports = aggregate_unnamed_reports(reports)
        self.result_data.append({
            'epoch': epoch,
            'mode': mode,
            'report': {k:all_reports[k].value() for k in all_reports}
        })
        if self.file_path:
            json.dump(self.result_data, open(self.file_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        logger.info('\n' + nice_report(all_reports))

    def reset_metrics(self):
        # rec
        self.rec_metrics.clear()
        # conv
        self.gen_metrics.clear()
        self.dist_set.clear()
        self.dist_ngram_cnt.clear()
        # optim
        self.optim_metrics.clear()
