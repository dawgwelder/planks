#!/usr/bin/python
# -*- coding: utf-8 -*-
from pprint import pprint
from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
import gridfs, os
import cv2
from sklearn import svm

import numpy as np
import pandas as pd

#/ creating connections for communicating with Mongo DB
client = MongoClient('localhost', 27017)
db = client.feat
fs = gridfs.GridFS(db)
fm = db.feats
data_dict = dict.fromkeys(['type', 'features'])

data = []

for gridout in fm.find({}):
    plist = gridout['features']

    for feats in plist:
        if ('knot' or 'сучок' or 'tar') in gridout['label']:
            data_dict['type'] = 'knot'
            data_dict['features'] = np.asarray(feats)
        else:
            data_dict['type'] = gridout['label']
            data_dict['features'] = np.asarray(feats)
        data.append(data_dict.copy())

length = len(data)
length_train = int(length//(10/7)) #data for train is 70% of data
                                   #data for test is 30%

data_train = data[:length_train]
data_test = data[length_train+1:]
X_train = []
y_train = []
X_test= []
y_test = []
for item in data_train:
    X_train.append(item['features'])
    y_train.append(item['type'])
for item in data_test:
    X_test.append(item['features'])
    y_test.append(item['type'])


lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
print (lin_clf.score(X_test, y_test))
result = lin_clf.predict(X_test)
print (np.mean(result == y_test))
for idx, item in enumerate(result):
    pprint (item + ' | ' + y_test[idx])




