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

import os
import json
import cPickle
from itertools import izip

import fire
import h5py
import tqdm
import numpy as np

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.np_utils import to_categorical

from misc import get_logger, Option
from network import TextOnly, top1_acc

opt = Option('../config.json')
cate1 = json.loads(open('./cate1.json').read())
DEV_DATA_LIST = ['../data/dev.chunk.01']
TEST_DATA_LIST = ['../data/test.chunk.01', '../data/test.chunk.02']


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.cate_type = 'bm'

    def get_sample_generator(self, ds, batch_size, div, raise_stop_event=False):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'image']]
          
            if div == 'train':
                if cate_type == 'bm':
                  Y = to_categorical(ds['cate'][left:right], 556)
                if cate_type == 's':
                  Y = to_categorical(ds['scate'][left:right], 3191)
                elif cate_type == 'd':
                  Y = to_categorical(ds['dcate'][left:right], 405)
            else:
                Y = None
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in cate1[d].iteritems()}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path,test_div, readable):
        pid_order = []

        if test_div == 'test':
            for data_path in TEST_DATA_LIST:
                h = h5py.File(data_path, 'r')[test_div]
                pid_order.extend(h['pid'][::])
        else:
            for data_path in DEV_DATA_LIST:
                h = h5py.File(data_path, 'r')[test_div]
                pid_order.extend(h['pid'][::])

        y2l = {i: s for s, i in meta['y_vocab'].iteritems()}
        y2l = map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0]))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        if cate_type == 'bm':
          for pid, y in izip(data['pid'], pred_y):
            label = y2l[y]
            tkns = list(map(int, label.split('>')))
            b, m = tkns
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            tpl = '{pid}\t{b}\t{m}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
            rets[pid] = tpl.format(pid=pid, b=b, m=m)
            no_answer = '{pid}\t-1\t-1'
        else:
          for pid, y in izip(data['pid'], pred_y):
              tpl = '{pid}\t{y}'
              rets[pid] = tpl.format(pid=pid, y=y)
          no_answer = '{pid}\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                ans = rets.get(pid, no_answer.format(pid=pid))
                print >> fout, ans

    def predict(self, data_root, model_root, test_root, test_div, out_path, cate_type_, readable=False):
        global cate_type
        cate_type = cate_type_

        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path).read())

        model_fname = os.path.join(model_root, 'model.h5')
        #self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model = load_model(model_fname,
                           custom_objects={'top1_acc': top1_acc})


        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')



        if test_div == 'train':
            test = test_data[test_div]
        else:
            test = test_data['dev']
        

        batch_size = opt.batch_size
        test_gen = self.get_sample_generator(test, batch_size, test_div, raise_stop_event=True)
        pred_y = []
        total_test_samples = test['uni'].shape[0]
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                _pred_y = model.predict(X)
                pred_y.extend([np.argmax(y) for y in _pred_y])
                pbar.update(X[0].shape[0])
        self.write_prediction_result(test, pred_y, meta, out_path, test_div, readable=readable)

    

    def train(self, data_root, out_dir, cate_type_='bm', is_validation = True):
        global cate_type
        cate_type = cate_type_

        if cate_type == 'bm':
            num_epochs = opt.bm_num_epochs
        elif cate_type == 's':
            num_epochs = opt.s_num_epochs
        elif cate_type == 'd':
            num_epochs = opt.d_num_epochs
        else:
            assert False, '%s is not valid data name' % cate_type
        
        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path).read())
        self.weight_fname = os.path.join(out_dir, 'weights')
        self.model_fname = os.path.join(out_dir, 'model')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['uni'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['uni'].shape[0])

        textonly = TextOnly()
        model = textonly.get_model(cate_type=cate_type)

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train,
                                              batch_size=opt.batch_size, div='train')
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))
        
        def schedule(epoch, lr):
            if cate_type == 'bm':
                if epoch == opt.bm_lr_change_epoch:
                    lr = 1e-5
            elif cate_type == 's':
                if epoch == opt.s_lr_change_epoch:
                    lr = 3e-5
            elif cate_type == 'd':
                if epoch == opt.d_lr_change_epoch:
                    lr = 3e-5
            return lr

        lrSchedule = LearningRateScheduler(schedule)

        if is_validation is True:
            checkpoint = ModelCheckpoint(self.weight_fname, monitor='val_loss',
                                         save_best_only=True, mode='min', period=1)

            total_dev_samples = dev['uni'].shape[0]
            dev_gen = self.get_sample_generator(dev,
                                                batch_size=opt.batch_size, div='train')
            self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

            model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=num_epochs,
                                validation_data=dev_gen,
                                validation_steps=self.validation_steps,
                                shuffle=True,
                                callbacks=[checkpoint, lrSchedule])
        else: 
            checkpoint = ModelCheckpoint(self.weight_fname, mode='min', period=1)

            model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=num_epochs,
                                validation_data=None,
                                shuffle=True,
                                callbacks=[checkpoint, lrSchedule])


            
        model.load_weights(os.path.join(out_dir, 'weights'))
        open(self.model_fname + '.json', 'w').write(model.to_json())
        model.save(self.model_fname + '.h5')

if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})
