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
        self._model = None

    def train_word2vec(self, num_features=50, min_word_count=1, context=10):
        """
        Train, saves or load exist Word2Vec model

        :param num_features    # Word vector dimensionality
        :param min_word_count  # Minimum word count
        :param context         # Context window size
        """
        model_path = "{f}features_{m}minwords_{c}context".format(f=num_features, m=min_word_count, c=context)
        model_path = os.path.join(self._output_path, model_path)
        if self._retrain is False and os.path.exists(model_path):
            self._model = word2vec.Word2Vec.load(model_path)
            print('[Word2Vec]Load existing model \'{m}\''.format(m=os.path.split(model_path)[-1]))
        else:
            # Set parameters
            num_workers = 2
            downsampling = 1e-3

            # Train the model
            print('[Word2Vec]Training model...')
            # sentences = [[vocabulary_inv[w] for w in s] for s in mtx_sentence]
            sentences = word2vec.Text8Corpus(self._corpus_path)
            self._model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

            # The model becomes effectively read-only, make the model much more memory-efficient.
            self._model.init_sims(replace=True)

            # Save the model
            print('[Word2Vec]Saving model \'{m}\''.format(m=os.path.split(model_path)[-1]))
            self._model.save(model_path)

    def get_embedding_weights(self):
        """
        Get initial weights for embedding layer.
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
