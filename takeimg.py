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
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

#/ creating connections for communicating with Mongo DB
client = MongoClient('localhost', 27017)
db = client.knots
fs = gridfs.GridFS(db)
fm = db.marked_up_image
data_dict = dict.fromkeys(['image', 'type', 'coordinates']) 
data = np.array([])
pand = pd.DataFrame()
pnd=[]

for gridout in fm.find({"users_polygons.polygons.type":{'$exists': True}}).limit(20):

    images = []
    
    plist = gridout['users_polygons']
    data_dict['image'] = gridout['image'] 
    for usrpoly in plist:
        poly = usrpoly['polygons']
        for p in poly:
            oxy = p['points']
            data_dict['type'] = p['type'] 
            for xy in oxy:
                pnd.append([xy['x'], xy['y']])
                data_dict['coordinates'] = pnd
                pand.concat(pand, pd.Series(data_dict))
        pnd = []

    try:
        with open('tmp.png', 'wb') as fi:
            fi.write(fs.get( ObjectId(gridout['image']) ).read() )
            img = cv2.imread('tmp.png')
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            ret, otsu = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
            vis = img.copy()
            mser = cv2.MSER_create()
            regions = mser.detectRegions(otsu, None)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            #cv2.imshow('img', vis)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            if img is not None:
                images.append(img)
                
                

                    
    except StopIteration:
        print("zero cursor")
    
                   
    
    

pprint (pand)
    
