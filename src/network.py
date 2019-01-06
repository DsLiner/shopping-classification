# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, concatenate
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from misc import get_logger, Option
opt = Option('../config.json')


def top1_acc(x, y):
  return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
  def __init__(self):
    self.logger = get_logger('model')

  def get_model(self, cate_type):
    if cate_type == 'bm':
      lr = opt.bm_lr
      embd_size = opt.bm_embd_size
      unigram_hash_size = opt.bm_unigram_hash_size
      image_fc_size = 1024
      hidden_size = 1024
      output_size = 556
    elif cate_type == 's':
      lr = opt.s_lr
      embd_size = opt.s_embd_size
      unigram_hash_size = opt.s_unigram_hash_size
      image_fc_size = 1024
      hidden_size = 1024
      output_size = 3191
    elif cate_type == 'd':
      lr = opt.d_lr
      embd_size = opt.d_embd_size
      unigram_hash_size = opt.d_unigram_hash_size
      image_fc_size = 512
      hidden_size = 512
      output_size = 405

    voca_size = unigram_hash_size + 1

    with tf.device('/gpu:0'):
      # input
      t_uni = Input((opt.max_len,), name='t_uni')
      w_uni = Input((opt.max_len,), name='w_uni')
      w_uni_mat = Reshape((opt.max_len, 1))(w_uni)
      image_feature = Input((opt.image_size,), name='image_feature')

      # Embedding
      t_uni_embd = Embedding(voca_size, embd_size)(t_uni)
      uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)      
      uni_embd = Reshape((embd_size,))(uni_embd_mat)
      embd_relu = Activation('relu')(uni_embd)

      # image
      image_fc = Dense(image_fc_size, activation='relu')(image_feature)

      #concate
      concat_wordImage = concatenate([embd_relu, image_fc], axis=1)
      FIANL_hidden = Dense(hidden_size, activation=None)(concat_wordImage)     
      dropout = Dropout(rate=0.5)(FIANL_hidden)
      relu = Activation('relu')(dropout)
      embd_image_out = Dense(output_size, activation='sigmoid', name='EmbdImage')(relu)

      # define Model
      model = Model(inputs=[t_uni, w_uni, image_feature],
                    outputs=embd_image_out)
      optm = keras.optimizers.Nadam(lr)
      model.load_weights('../data/model/s/weights')
      model.compile(loss='binary_crossentropy',
                    optimizer=optm,
                    metrics=[top1_acc])
      model.summary(print_fn=lambda x: self.logger.info(x))

    return model