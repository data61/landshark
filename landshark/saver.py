import os
from copy import deepcopy
import shutil
from glob import glob
import logging

import json

from typing import Dict
import numpy as np

log = logging.getLogger(__name__)


def overwrite_model_dir(model_dir: str, checkpoint_dir: str) -> None:
    """Copy the checkpoints from their directory into the model dir."""
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(checkpoint_dir, model_dir)

class BestScoreSaver:
    """Saver for only saving the best model based on held out score.

    This now persists between runs by keeping a JSON file in the model
    directory.
    """

    def __init__(self, directory: str) -> None:
        """Saver initialiser."""
        self.directory = directory

    def _init_dir(self, score_path: str) -> None:
        if not os.path.exists(score_path):
            os.mkdir(score_path)
            shutil.copy2(os.path.join(self.directory, "METADATA.bin"),
                         score_path)

    def _to_64bit(self, scores: Dict[str, np.ndarray]) \
            -> Dict[str, np.ndarray]:
        new_scores = deepcopy(scores)
        # convert scores to 64bit
        for k, v in new_scores.items():
            if v.dtype == np.float32:
                new_scores[k] = v.astype(np.float64)
            if v.dtype == np.int32:
                new_scores[k] = v.astype(np.int64)
        return new_scores


    def _should_overwrite(self, s: str, score: np.ndarray,
                          score_path: str) -> bool:
        score_file = os.path.join(score_path, "model_best.json")
        overwrite = True
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                best_scores = json.load(f)
            if s == "loss":
                if best_scores[s] < score:
                    overwrite = False
            else:
                if best_scores[s] > score:
                    overwrite = False
        return overwrite


    def _write_score(self, scores: Dict[str, np.ndarray],
                     score_path: str, global_step: int) -> None:
        score_file = os.path.join(score_path, "model_best.json")
        with open(score_file, 'w') as f:
            json.dump(scores, f)
        checkpoint_files = glob(os.path.join(self.directory,
                                "model.ckpt-{}*".format(global_step)))
        deleting_files = glob(os.path.join(score_path, "model.ckpt-*"))
        for d in deleting_files:
            os.remove(d)
        for c in checkpoint_files:
            shutil.copy2(c, score_path)

    def save(self, scores: dict) -> None:
        scores = self._to_64bit(scores)
        global_step = scores.pop("global_step")
        # Create directories if they don't exist
        for s in scores.keys():
            score_path = self.directory + "_best_{}".format(s)
            self._init_dir(score_path)
            if self._should_overwrite(s, scores[s], score_path):
                log.info("Found model with new best {} score: overwriting".
                         format(s))
                self._write_score(scores, score_path, global_step)

