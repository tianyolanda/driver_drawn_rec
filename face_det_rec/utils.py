import os
import numpy as np
import time

def get_filenames(rootdir):
    image_registration_dir = rootdir + 'image_registration/'

    image_registration_img_dir = image_registration_dir + 'image/'

    files_path = []

    for root,dirs,files in os.walk(rootdir,topdown=True):
        for name in files:
            _,encoding = os.path.splitext(name)
            print(_)
            if encoding == ".jpg":
                files_path.append(os.path.join(root,name))
    return files_path


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes. boxes in corner-form
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.

    usage: maybe n boxes
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
    """
    # print('boxes0,boxes1',boxes0,boxes1)
    boxes0 = np.array(boxes0)
    boxes1 = np.array(boxes1)
    # print('boxes0[..., :2], boxes1[..., :2]',boxes0[..., :2], boxes1[..., :2])
    overlap_left_top = np.maximum(boxes0[:, :2], boxes1[:, :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

# class Person:
#     def __init__(self, id, type, status, face_feature):
#         self.id = id
#         self.type = type  # family
#         self.status = status  # out in
#         self.face_feature = face_feature
#         self.counter = 0
#     def change_status(self):
#         if self.status == 0:
#             self.status = 1
#         elif self.status == 1:
#             self.status = 0
#
# class DetectedFace:
#     def __init__(self, location_in_image):
#         self.has_ID = False
#         self.location_in_image = location_in_image # in cornor
#         self.color = (0, 0, 255)  # red
#         self.tracked = False
#         self.tracking_frames = 8  # how many consecutive frames has this person
#         self.lost_frames = 0  # how many frames haven't seen this person
#         self.person = 'unknown'
#
#     def setID(self, person):
#         self.has_ID = True
#         self.person = person
#         self.color = (0, 255, 0)  # green
#         self.person.counter = self.tracking_frames  # 需要判断下深浅拷贝
#
#     def tracking(self, location):
#         self.tracked = True
#         self.tracking_frames += 1
#         self.location_in_image = location
#         self.person.counter = self.tracking_frames
#
# person = Person(0,0,0,0)
# print(person.counter)
#
# detf = DetectedFace(100)
#
# detf.setID(person)
# print('person.counter, detf.tracking_frames:',person.counter, detf.tracking_frames)
#
# detf.tracking(200)
#
# print('person.counter, detf.tracking_frames:',person.counter, detf.tracking_frames)
# detf.person.change_status()

# a = [2,3,4,5,6]
# for i in a:
#     print(i)


