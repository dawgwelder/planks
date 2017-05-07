#!/usr/bin/python
# -*- coding: utf-8 -*-
from pprint import pprint
from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
import gridfs, os
import collections
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# / creating connections for communicating with Mongo DB
client = MongoClient('localhost', 27017)
db = client.knots
fs = gridfs.GridFS(db)
fm = db.marked_up_image

data_dict = dict.fromkeys(['image', 'type', 'coordinates'])

data = []

for gridout in fm.find({"users_polygons.polygons.type":{"$exists":True}}).limit(20):

    images = []
    plist = gridout['users_polygons']

    data_dict['image'] = gridout['image']
    for usrpoly in plist:
        poly = usrpoly['polygons']
        data_dict['coordinates'] = []
        for p in poly:
            #pprint (p)
            points = p['points']
            data_dict['type'] = p['type']
            tmp = []
            for point in points:
                #pprint (point)
                tmp.append((point['x'], point['y']))
            data_dict['coordinates'] = np.asarray([tmp], dtype=np.int32)
            #pprint (data_dict)
            data.append(data_dict.copy())
            #pprint (data)

for defect in data:
    pprint ((defect['coordinates']))
    if defect['type'] == 'knot_defect':
        try:
            with open('tmp.png', 'wb') as fi:
                fi.write(fs.get(ObjectId(defect['image'])).read())
                img = cv2.imread('tmp.png', -1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                mask = np.zeros(img.shape, dtype=np.uint8)
                roi_corners = defect['coordinates']
                # fill the ROI so it doesn't get wiped out when the mask is applied
                channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,) * channel_count
                cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                # from Masterfool: use cv2.fillConvexPoly if you know it's convex

                # apply the mask
                masked_image = cv2.bitwise_and(img, mask)

                cv2.imshow('img', masked_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                if img is not None:
                    images.append(img)




        except StopIteration:
            print("zero cursor")
            # data.append(data_dict)


