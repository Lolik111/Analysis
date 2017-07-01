import numpy as np
import cv2
import dlib

from CV.Tracker import Tracker
import CV.BlurProcessing

num_cascade = cv2.CascadeClassifier(r'C:\opencv\build\etc\haarcascades\haarcascade_russian_plate_number.xml')
cap = cv2.VideoCapture('video3.avi')
# mog = cv2.createBackgroundSubtractorMOG2()

trackers = []
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(WIDTH)
print(HEIGHT)
i = 0
# bounds
LEFT_BOUND = 0
RIGHT_BOUND = 1800
TOP_BOUND = 500
BOTTOM_BOUND = -1


def bb_intersection_over_union(recta, rectb):
    """
 determine the (x, y)-coordinates of the intersection rectangle
    :param boxA:
    :param boxB:
    :return:
    """
    boxA = [recta[0], recta[1], recta[0] + recta[2], recta[1] + recta[3]]
    boxB = [rectb[0], rectb[1], rectb[0] + rectb[2], rectb[1] + rectb[3]]

    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def intersection(recta, rectb):
    boxA = [recta[0], recta[1], recta[0] + recta[2], recta[1] + recta[3]]
    boxB = [rectb[0], rectb[1], rectb[0] + rectb[2], rectb[1] + rectb[3]]

    if boxA[0] > boxB[2] or boxA[2] < boxB[0]:
        return 0
    if boxA[1] > boxB[3] or boxA[3] < boxB[1]:
        return 0
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
    return interArea

def cv_size(img):
    return tuple(img.shape[1::-1])


def explicit_max(l):
    max_val = max(l)
    max_idx = l.index(max_val)
    return max_idx, int(max_val)



def detect_over_exposed(img):
    histr = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    return np.sum(histr[0:210]) < np.sum(histr[210:-1]) * 2


def handle_tracker_data(t_tracker):
    if len(t_tracker.data) <= 1:
        return
    exposion_treshold = 1.4
    for j, image in enumerate(t_tracker.data):
        (angle, length) = t_tracker.short_estimate_mov(j)
        var_lap = CV.BlurProcessing.varianceOfLaplacian(image)
        tenen = CV.BlurProcessing.tenengrad(image)
        canny = CV.BlurProcessing.canny_area_dependent_method(image)
        exposion = CV.BlurProcessing.detect_exposion_level(image)
        cv2.imwrite(r'lure\z' + str(t_tracker.index) + '-' + str(j) + '-' + str(var_lap // 1) + '-' + str(tenen // 1) + '-' + str(canny // 1) + '-' + str(exposion) + '.png', image)
        # cv2.imwrite(r'lure\z' + str(t_tracker.index) + '-' + str(j) + '-' + str(angle) + '-' + str(length) + '.png', image)
    """var_lap = []
    tenen = []
    canny_level = []
    exposion_level = []
    for j, image in enumerate(t_tracker.data):
        var_lap.append(CV.BlurProcessing.varianceOfLaplacian(image))
        tenen.append(CV.BlurProcessing.tenengrad(image))
        canny_level.append(CV.BlurProcessing.canny_area_dependent_method(image))
        exposion_level.append(CV.BlurProcessing.detect_exposion_level(image))
    idx1, val1 = explicit_max(var_lap)
    idx2, val2 = explicit_max(tenen)
    idx3, val3 = explicit_max(canny_level)
    idx4, val4 = explicit_max(exposion_level)

    cv2.imwrite(r'blur\r' + str(t_tracker.index) + '-var_lap-' + str(idx1) + '-' + str(val1) + '.png', t_tracker.data[idx1])
    cv2.imwrite(r'blur\r' + str(t_tracker.index) + '-tenen-' + str(idx2) + '-' + str(val2) + '.png', t_tracker.data[idx2])
    cv2.imwrite(r'blur\r' + str(t_tracker.index) + '-canny-' + str(idx3) + '-' + str(val3) + '.png', t_tracker.data[idx3])
    cv2.imwrite(r'blur\r' + str(t_tracker.index) + '-exposion-' + str(idx4) + '-' + str(val4) + '.png', t_tracker.data[idx4])
    """
    return


ind = 0
while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret:
        break
    # bounded_frame = frame[TOP_BOUND:BOTTOM_BOUND, LEFT_BOUND:RIGHT_BOUND]
    bounded_frame = cv2.resize(frame[TOP_BOUND:BOTTOM_BOUND, LEFT_BOUND:RIGHT_BOUND], dsize=None, fx=0.5, fy=0.5)
    original = frame[TOP_BOUND:BOTTOM_BOUND, LEFT_BOUND:RIGHT_BOUND].copy()
    gray = cv2.cvtColor(bounded_frame, cv2.COLOR_BGR2GRAY)

    # if i % 5 != 0:
    #     continue
    numbers = num_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 20), maxSize=(215, 75))
    new_trackers = []
    areas = []
    trackers_used = []
    for (x, y, w, h) in numbers:
        cv2.imwrite('trackimg\\rect' + str(i) + '.png', bounded_frame[y: y + h, x: x + w])
        cv2.rectangle(bounded_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for tracker in trackers:
        trackingQuality, (x, y, w, h) = tracker.update(bounded_frame)
        # print(trackingQuality)
        cv2.rectangle(bounded_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        areas.append((x, y, w, h))

        trackers_used.append(False)

    for (x, y, w, h) in numbers:
        # cv2.rectangle(bounded_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        index = -1
        for j, area in enumerate(areas):
            if intersection((x, y, w, h), area) != 0:
                index = j
                trackers_used[j] = True
                break
        if index >= 0:
            trackers[index].append_data(original, (x, y, w, h))
        else:
            new_tracker = Tracker("dlib")
            new_tracker.init(bounded_frame, (x - 20, y - 20, w + 40, h + 20))
            new_tracker.append_data(original, (x, y, w, h))
            new_trackers.append(new_tracker)
            # cv2.imwrite('trackimg\\rect' + str(i) + '.png', bounded_frame[y: y + h, x: x + w])

    temp_trackers = []
    for j, flag in enumerate(trackers_used):
        if flag:
            temp_trackers.append(trackers[j])
        else:
            handle_tracker_data(trackers[j])

    trackers = temp_trackers
    print(len(numbers), len(trackers), len(new_trackers))
    # temp_trackers.clear()
    trackers.extend(new_trackers)

    res = bounded_frame
    # res = cv2.resize(bounded_frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('img', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
