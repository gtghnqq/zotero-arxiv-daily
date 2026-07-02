from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
import os

from sentence_transformers import SentenceTransformer

# 全局缓存模型
_encoder = None


@register_reranker("local")
class LocalReranker(BaseReranker):

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        global _encoder

        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        # 只加载一次模型
        if _encoder is None:
            _encoder = SentenceTransformer(
                self.config.reranker.local.model,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")  # 如果配置了 HF_TOKEN
            )

        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}

        s1_feature = _encoder.encode(
            s1,
            **encode_kwargs,
            show_progress_bar=True
        )
        s2_feature = _encoder.encode(
            s2,
            **encode_kwargs,
            show_progress_bar=True
        )

        sim = _encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
