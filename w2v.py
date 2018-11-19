import os
import numpy as np
from gensim.models import word2vec


class Word2Vec:
    def __init__(self, corpus_path, dict_i2w, retrain=True):
        """
        :param corpus_path: crop file(.txt) to train
        :param dict_i2w: dict {index(int), word(str)}
        :param retrain: retrain the model or load the exist model
        """
        self._corpus_path = corpus_path
        self._dict_i2w = dict_i2w
        self._retrain = retrain
        self._output_path = os.path.join(os.getcwd(), 'model_w2v')
        if not os.path.exists(self._output_path):
            os.mkdir(self._output_path)
        self._model = self.train_word2vec()

    def train_word2vec(self):
        """
        Train, saves or load exist Word2Vec model
        """
        # Set parameters
        workers = 2  # 執行緒數量
        size = 256  # 訓練維度(維度太小將導致無法有效表示詞與詞之間的關係、維度太大會導致關係稀疏難以找出規則)
        min_count = 1  # 詞數小於這個值，不被視為訓練對象
        window = 5  # 詞向量上下文最大距離。cbow: 決定取多少詞來預測中間詞; skip-gram: 反之
        # sg: 1=skip-gram,對低頻詞敏感; 0=cbow(default)
        # iter: 訓練回數

        model_path = "{s}size_{m}mincount_{w}window".format(s=size, m=min_count, w=window)
        model_path = os.path.join(self._output_path, model_path)
        if self._retrain is False and os.path.exists(model_path):
            model = word2vec.Word2Vec.load(model_path)
            print('[Word2Vec]Load existing model \'{m}\''.format(m=os.path.split(model_path)[-1]))
        else:
            # Train the model
            print('[Word2Vec]Training model...')
            # sentences = [[vocabulary_inv[w] for w in s] for s in mtx_sentence]
            sentences = word2vec.Text8Corpus(self._corpus_path)
            model = word2vec.Word2Vec(sentences=sentences, workers=workers, size=size, min_count=min_count,
                                      window=window)

            # The model becomes effectively read-only, make the model much more memory-efficient.
            model.init_sims(replace=True)

            # Save the model
            print('[Word2Vec]Saving model \'{m}\''.format(m=os.path.split(model_path)[-1]))
            model.save(model_path)
        return model

    def get_embedding_weights(self):
        """
        Get initial weights for embedding layer
        """
        if self._model is None:
            return None
        else:
            embedding_weights = {}
            for key, word in self._dict_i2w.items():
                if word in self._model:
                    embedding_weights[key] = self._model[word]
                else:
                    embedding_weights[key] = np.random.uniform(-0.25, 0.25, self._model.vector_size)  # unknown words
            return embedding_weights
