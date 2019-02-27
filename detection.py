rom Detection.Base import DetectionBase
import logging
from PIL import Image
from Detection.config.yolo_config import *
from Detection.models.yolo2 import YOLO

import time

logging.basicConfig(format='[%(levelname)s|%(asctime)s] %(message)s',
datefmt='%Y%m%d %H:%M:%S',
level=logging.DEBUG)


class Detection(DetectionBase):
"""
Using YOLO version
"""
def __init__(self, image_id, image_src, cam_id):

super(Detection, self).__init__(image_id, image_src, cam_id)
self._img = self.image['src']
self._result = {}
self.model = YOLO(WEIGHTS,classes_path=CLASSES_PATH, anchors_path=ANCHORS_PATH)
if self._img.mode != "RGB":
self._img = self.image.convert("RGB")

def getOutput_cropped(self, threshold = 0.1):
"""

:param model:
:param image:
:param threshold:
:return: list of cropped images
"""
output_image = []
start_time = time.time()
results = []
try:
results = self.model.detect(self._img)
logging.info("Detection Image. Time: {}".format(time.time() - start_time))
except Exception as ex:
logging.error("Detect Image error: {}".format(ex))

self._result['cam_id'] = self.cam_id
self._result['image_id'] = self.image["_id"]
self._result['image_src'] = self.image["src"]

img = {}
for i, re in enumerate(results):
im_tag = re[0].split(' ')[0]
if im_tag == 'satudora_product':
if float(re[0].split(' ')[1]) > threshold:
image_cropped = self._img.crop(re[1:])
img['box_id'] = i
img['cropped_img'] = image_cropped
output_image.append(img)
self._result['cropped_image'] = output_image

def getOutput(self, threshold=0.1):
"""
boxes: array of objects
box_id: the id of bounding box
detection: object
box_geometry: [coordinates]
confidence: the result of detection model
label: the label of object
:param model:
:param image:
:param threshold:
:return:
"""
boxes = []

start_time = time.time()
results = []
try:
results = self.model.detect(self._img)
logging.info("Detection Image. Time: {}".format(time.time() - start_time))
except Exception as ex:
logging.error("Detect Image error: {}".format(ex))

self._result['cam_id'] = self.cam_id
self._result['image_id'] = self.image["_id"]
self._result['image_src'] = self.image["src"]

for i, re in enumerate(results):
im_tag = re[0].split(' ')[0]
box = {}
if im_tag == 'satudora_product':
if float(re[0].split(' ')[1]) > threshold:
box['box_id'] = i
box['coordinates'] = re[1:]
box['confidence'] = float(re[0].split(' ')[1])
box['label'] = im_tag
boxes.append(box)
self._result['boxes'] = boxes