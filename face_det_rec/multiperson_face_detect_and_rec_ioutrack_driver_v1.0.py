import argparse
import numpy as np
import cv2 as cv
import os
import math
import datetime
import yaml
import copy
from easydict import EasyDict as edict
from utils import str2bool, area_of,iou_of

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Message:
    def __init__(self, root_dir):
        self.terminal_number = 1
        self.message_number = 0
        self.lines_number = 0
        self.root_dir = root_dir + 'message_to_AI/'

    def pub_to_txt(self, report_type, img, q, status):
        current_time = datetime.datetime.now()
        current_time_format = current_time.strftime("%Y-%m-%d_%H-%M-%S_%f")  # keep the time: year,month,day,hour,mintue,second,microsecond

        image_file_name = str(current_time_format) + '.jpg'
        image_file_path = self.root_dir + 'image/' + image_file_name
        cv.imwrite(image_file_path, img)

        # create txt file
        m = str(self.terminal_number)
        nnn = str(self.message_number).zfill(3)
        txt_file_name = 'msg_' + m + '_' + nnn + '.txt'
        txt_file_path = self.root_dir + 'txt/' + txt_file_name
        file = open(txt_file_path, mode='a') # a: add into the txt

        # write to txt file
        r = ['s', 'n', 'c', 'a']  # report type
        f = ['i', 'o', 'u']  # for unknown people and guest, set 'u' (don't know in or out)

        write_content = m + ' ' + nnn + ' ' + r[report_type] + ' ' + image_file_name + ' ' + str(q) + ' ' + f[status] +'\n'
        file.writelines(write_content)
        file.close()

        self.lines_number += 1
        if self.lines_number > 10000:
            self.message_number += 1
            self.lines_numer = 0

class Person:
    def __init__(self, id, type, face_feature):
        self.id = id
        self.type = type  # family
        # self.status = status  # out in
        self.face_feature = face_feature
        self.lost_counter = 0
        self.tracking_counter = 0
        self.appeared = False
        self.message_sent = False
        self.det_confidene = 0
        self.rec_confidence = 0
        self.appeared_image = None

    def print_in_terminal(self):
        person_type = ['Driver', 'Drivers Family', 'Guest   ']
        print(person_type[self.type],'detected.')

            
    def calculate_confidence_for_registed_people(self, detection_threshold, tracking_threshold, cosine_similarity_threshold):
        # consist of three confidence: detection, recognition, tracking. Sum them as the overall confidence
        weight = [1,1,1] # weight of detection, recognition, tracking.

        detection_confidence = (self.det_confidene - detection_threshold)/(1-detection_threshold) * 5

        keep_tracking_confidence = 5

        recognition_confidence = min(self.rec_confidence - cosine_similarity_threshold, 1 - cosine_similarity_threshold) / (1 - cosine_similarity_threshold) * 5 # 0~5

        overall_confidence = (detection_confidence * weight[0] + recognition_confidence * weight[1] + keep_tracking_confidence * weight[2]) / sum(weight)

        overall_confidence = math.ceil(overall_confidence)  # upper level

        return overall_confidence

class DetectedFace:
    def __init__(self, location_in_image, det_conf):
        person_init = Person(-1,-1,-1)

        self.has_ID = False
        self.location_in_image = location_in_image # in cornor
        self.color = (0, 0, 255)  # red
        self.tracked = False
        self.tracking_frames = 0  # how many consecutive frames has this person
        self.person = person_init
        self.det_confidene = det_conf
        self.rec_confidence = 0
        self.face_image = None
        self.unknown_rec_confidence_list = []

    def setID(self, p, rec_conf, image):
        self.has_ID = True
        self.rec_confidence = rec_conf
        self.color = (0, 255, 0)  # green
        self.face_image = image

        self.person = p
        self.person.tracking_counter = self.tracking_frames
        self.person.lost_counter = 0
        self.person.appeared = True
        self.person.rec_confidence = self.rec_confidence
        self.person.det_confidene = self.det_confidene
        self.person.appeared_image = self.face_image

    def tracking(self, location):
        self.tracked = True
        self.tracking_frames += 1
        self.location_in_image = location
        if self.has_ID:
            self.person.tracking_counter = self.tracking_frames
            self.person.lost_counter = 0
    def calculate_avg_rec_conf_for_unknown_people(self):
        sum = 0
        for item in self.unknown_rec_confidence_list:
            sum += item
        self.rec_confidence =  sum / len(self.unknown_rec_confidence_list)

    def calculate_confidence_for_unknown_people(self, detection_threshold, tracking_threshold, cosine_similarity_threshold):
        # consist of three confidence: detection, recognition, tracking. Sum them as the overall confidence

        weight = [1,2,3] # weight of detection, recognition, tracking.

        detection_confidence = (self.det_confidene - detection_threshold)/(1-detection_threshold) * 5

        keep_tracking_confidence = min(self.tracking_frames / tracking_threshold, 5)  # 1~5
        recognition_confidence = (cosine_similarity_threshold - self.rec_confidence) / (1 - cosine_similarity_threshold) * 5

        overall_confidence = (detection_confidence * weight[0] + recognition_confidence * weight[1] + keep_tracking_confidence * weight[2]) / sum(weight)
        overall_confidence = math.ceil(overall_confidence)  # upper level
        return overall_confidence
        # self.det_confidene

def visualize(input, faces, fps, scale, thickness=2):
    property_class = ['driver', 'drivers family', 'guest']
    if len(faces) != 0:
        for face in faces:
            # scaling is only applied in visualization, do not change face target, so use deep copy
            coords_float = copy.deepcopy(face.location_in_image)  # [left, top, right, bottom]
            coords_float[0] *= 1/scale
            coords_float[1] *= 1/scale
            coords_float[2] *= 1/scale
            coords_float[3] *= 1/scale

            coords = list(map(int, coords_float))

            cv.rectangle(input, (coords[1], coords[0]), (coords[3], coords[2]), face.color, thickness)

            font = cv.FONT_HERSHEY_DUPLEX
            if face.has_ID:
                type = face.person.type
                name = property_class[type]
            else:
                name = 'unknown'
            cv.putText(input, name, (coords[1], coords[2]), font, 1.0, face.color, 2)
            cv.putText(input, 'tracked:'+ str(face.tracking_frames), (coords[1], coords[2]+20), font, 1.0, (0,255,255), 1)  # yellow
            cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def registration_face_feature_extraction(detector, recognizer, rootdir, image_scale):

    image_registration_dir = rootdir + 'image_registration/'

    image_registration_img_dir = image_registration_dir + 'image/'

    registed_person_infos = []

    for root, dirs, files in os.walk(image_registration_img_dir, topdown=True):
        for name in files:
            id, encoding = os.path.splitext(name)
            if encoding == ".jpg":
                image_path = os.path.join(root,name)
                img = cv.imread(cv.samples.findFile(image_path))
                imgWidth = int(img.shape[1] * image_scale)
                imgHeight = int(img.shape[0] * image_scale)

                img = cv.resize(img, (imgWidth, imgHeight))

                ## [inference]
                # Set input size before inference
                detector.setInputSize((imgWidth, imgHeight))

                face = detector.detect(img)

                face_align = recognizer.alignCrop(img, face[1][0])

                face_feature = recognizer.feature(face_align)

                txt_path = image_registration_dir + 'property/' + id + '.txt'
                f = open(txt_path)
                type = f.readline().strip('\n')  # line1
                # status = f.readline().strip('\n')  # line2

                person = Person(id, int(type),  face_feature)
                registed_person_infos.append(person)

    return registed_person_infos


if __name__ == '__main__':
    # read parameters
    parameter_file = 'parameters.yaml'
    with open(parameter_file, 'r') as f:
        config = edict(yaml.safe_load(f))

    # initialize timer
    tm = cv.TickMeter()

    # initialize message
    msg = Message(config.PATH.root_dir)

    # [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        config.PATH.face_detection_model,
        "",
        (320, 320),
        config.DETECTION.score_threshold,
        config.DETECTION.nms_threshold,
        config.DETECTION.nms_top_k
    )

    ## [initialize_Face Recognizer]

    recognizer = cv.FaceRecognizerSF.create(
        config.PATH.face_recognition_model, "")

    # extract the features of all registration faces, from image_registration dir
    registration_person_infos = registration_face_feature_extraction(detector, recognizer, config.PATH.root_dir, config.IMAGE.scale)

    # initialize camera information
    # cap = cv.VideoCapture(0) # 0: real-time. you can read a video by changing 0 to video file path
    video_file = config.PATH.video_file
    cap = cv.VideoCapture(video_file) # read a video by path

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)* config.IMAGE.scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)* config.IMAGE.scale)
    detector.setInputSize([frameWidth, frameHeight])

    # initialize iou tracker
    frame_counter = 0
    faces_in_lastframe = []

    if config.OUTPUT.save is True:
        video2save_path = './test_videos/output/'+ 'det_thr'+ str(config.DETECTION.score_threshold) + '_cs_thr' + str(config.RECOGNITION.cosine_similarity_threshold_medium) + '_iou_thr' + str(
            config.TRACKING.iou_threshold) + '_' + str(frameWidth) + '*' + str(frameHeight) + '.mp4'

        camera_rate = {'1-logi-c270n': [27],
                       '2-elp': [29.97],
                       '3-buffalo': [28],
                       '4-logi-c615n': [30]}

        # camera_rate_size = {'1-logi-c270n': [27, (1280, 852)],
        #                     '2-elp': [29.97, (640, 426)],
        #                     '3-buffalo': [28, (1620, 1080)],
        #                     '4-logi-c615n': [30, (1620, 1080)]}
        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        print('output will save in:', video2save_path)
        output_movie = cv.VideoWriter(video2save_path, fourcc, 30,
                                      (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        tm.start()
        # print('---------------------------new frame')

        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame_ori = frame
        frame = cv.resize(frame, (frameWidth, frameHeight))

        # detection: find all faces in current frame
        faces = detector.detect(frame)  # faces is a tuple

        faces_in_thisframe = []

        if len(faces_in_lastframe) != 0:
            for faceInLastFrame in faces_in_lastframe:
                faceInLastFrame.tracked = False

        if faces[1] is not None:  # if this frame has face
            for idx, face in enumerate(faces[1]): # loop every faces in this frame
                detectedFace = DetectedFace([face[1], face[0], face[1]+face[3], face[0]+face[2]],face[-1])
                ious = []

                # tracking with faces of lastframe

                if len(faces_in_lastframe) != 0:
                    for faceInLastFrame in faces_in_lastframe:
                        iou = iou_of([detectedFace.location_in_image], [faceInLastFrame.location_in_image])  
                        ious.append(iou)
                    ious = np.array(ious)
                    iou_best_match_index = np.argmax(ious)  # find the best match with highest last frame bbox by iou, return index
                    max_iou = ious[iou_best_match_index]
                    if max_iou > config.TRACKING.iou_threshold:
                        faces_in_lastframe[iou_best_match_index].tracking(detectedFace.location_in_image)
                        detectedFace = faces_in_lastframe[iou_best_match_index]

                if detectedFace.has_ID is False:
                    face_align = recognizer.alignCrop(frame, face)

                    face_feature = recognizer.feature(face_align)

                    # recognition: find the most matching person from registration info
                    face_distances = []
                    for person in registration_person_infos:
                        cosine_score = recognizer.match(person.face_feature, face_feature, cv.FaceRecognizerSF_FR_COSINE)
                        face_distances.append(cosine_score)
                    
                    best_match_index = np.argmax(face_distances)  # find the one with highest cosine score. # if empty sequence error: check the root_dir in parameters.yaml 
                    max_score = face_distances[best_match_index]

                    if max_score >= config.RECOGNITION.cosine_similarity_threshold_high:  # one frame has very high similarity -> change to registered person
                        detectedFace.setID(registration_person_infos[best_match_index], max_score, frame)
                    elif max_score >= config.RECOGNITION.cosine_similarity_threshold_medium and detectedFace.rec_confidence > config.RECOGNITION.unknown_person_avg_rec_thresh:  # sequence frame has less similarity, but successful tracking for a while -> then change to registered person
                        detectedFace.setID(registration_person_infos[best_match_index], max_score, frame)
                    else:  # unknown person
                        detectedFace.unknown_rec_confidence_list.append(max_score)
                        detectedFace.calculate_avg_rec_conf_for_unknown_people()
                        detectedFace.face_image = frame

                faces_in_thisframe.append(detectedFace)

        #  clear cache: the unknown peole who was detected in last frame but not appeared in this frame -> send warning message
        if len(faces_in_lastframe) != 0:
            for faceInLastFrame in faces_in_lastframe:
                if faceInLastFrame.tracked == False: # this frame no this person
                    if (not faceInLastFrame.has_ID) and (faceInLastFrame.tracking_frames > config.TRACKING.unknown_person_tracking_thresh) and (faceInLastFrame.rec_confidence < config.RECOGNITION.unknown_person_avg_rec_thresh):
                        conf = faceInLastFrame.calculate_confidence_for_unknown_people(config.DETECTION.score_threshold, config.TRACKING.tracking_thresh, config.RECOGNITION.cosine_similarity_threshold_medium)

                        print('Alert!     Unknown person appears at car.')

                        # status = 2  # unknown people: 'unknown' he is coming in or going out
                        # msg.pub_to_txt(report_type, faceInLastFrame.face_image, conf, status)

        # clear cache: the registration people who appeared in recent, but has lost for a while -> change inhome/outhome status
        for registration_person in registration_person_infos:
            if registration_person.appeared:
                if (registration_person.tracking_counter > config.TRACKING.tracking_thresh) and (not registration_person.message_sent):
                    conf = registration_person.calculate_confidence_for_registed_people(config.DETECTION.score_threshold, config.TRACKING.tracking_thresh, config.RECOGNITION.cosine_similarity_threshold_medium)
                    registration_person.print_in_terminal()
                    registration_person.message_sent = True

                # every frame add the lost counter. If this person has been tracked, this number will be set to 0 in other fuctions, e.g., tracking(), setID().
                registration_person.lost_counter += 1

                # if this person has lost for a while, we assume he has left the door region, so reset his informations, preparing for his next coming.
                if registration_person.lost_counter > config.TRACKING.known_people_lost_thresh:
                    registration_person.appeared = False
                    registration_person.message_sent = False

        faces_in_lastframe = faces_in_thisframe

        tm.stop()  # record fps

        # Draw results on the rescaled input image
        visualize(frame_ori, faces_in_thisframe, tm.getFPS(), config.IMAGE.scale)
        #visualize(frame, faces_in_thisframe, tm.getFPS(), 1)

        # Visualize results
        cv.imshow('Live', frame_ori)

        if config.OUTPUT.save:
            output_movie.write(frame_ori)

    cv.destroyAllWindows()

