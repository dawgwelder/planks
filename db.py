#!/usr/bin/python
# -*- coding: utf-8 -*-

from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
import gridfs, os

#/ creating connections for communicating with Mongo DB
client = MongoClient('localhost', 27017)
db = client.knots
fs = gridfs.GridFS(db)

if fs.exists(ObjectId("5787a46e51208b00287feb4a")): 
    print ('yup')
else:
    print ('nope')

with open('testknot.png', 'wb') as fi:
    fi.write(fs.get(ObjectId("5787a46e51208b00287feb4a")).read())


