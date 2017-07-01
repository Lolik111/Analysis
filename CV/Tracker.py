import dlib
import cv2
import math


class Tracker:
    """
    Tracker
    """
    private_ind = 0

    def __init__(self, ttype):
        if ttype == "dlib":
            self.tracker = dlib.correlation_tracker()
            self.ttype = 1
        elif ttype in ["BOOSTING", "MIL", "TLD", "MEDIANFLOW"]:
            self.tracker = cv2.Tracker_create(ttype)
            self.ttype = 2
        else:
            raise ValueError("Wrong type of tracker")
        # self.previous_rect = None
        # self.current_rect = None
        self.data = []
        self.rectangles = []
        self.index = Tracker.private_ind
        Tracker.private_ind += 1

    def init(self, frame, rectangle):
        if self.ttype == 1:
            self.tracker.start_track(frame, dlib.rectangle(int(rectangle[0]), int(rectangle[1]),
                                                           int(rectangle[0] + rectangle[2]),
                                                           int(rectangle[1] + rectangle[3])))
        else:
            self.tracker.init(frame, rectangle)
            # self.current_rect = rectangle

    def update(self, frame):
        if self.ttype == 1:
            tracking_quality = self.tracker.update(frame)
            tracked_position = self.tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            bbox = [t_x, t_y, t_w, t_h]
        else:
            ok, bbox = self.tracker.update(frame)
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            tracking_quality = 10 if ok else 0
        # self.previous_rect = self.current_rect
        # self.current_rect = bbox
        return tracking_quality, bbox

    def append_data(self, frame, rectangle):
        (x, y, w, h) = rectangle
        self.data.append(frame[y * 2:y * 2 + h * 2, x * 2: x * 2 + w * 2].copy())
        self.rectangles.append(rectangle)

    def estimate_mov(self):
        for j, rect in enumerate(self.rectangles):
            if j == 0: continue
            (p_x, p_y, p_w, p_h) = self.rectangles[j - 1]
            (x, y, w, h) = self.rectangles[j]

    def short_estimate_mov(self, i):
        z = 1 if i == 0 else i
        (p_x, p_y, p_w, p_h) = self.rectangles[z - 1]
        (x, y, w, h) = self.rectangles[z]
        (p_c_x, p_c_y) = (p_x + p_w/2, p_y + p_h/2)
        (c_x, c_y) = (x + w/2, y + h/2)
        length = math.sqrt(math.fabs(p_c_x - c_x)**2 + math.fabs(p_c_y - c_y)**2)
        angle = math.acos((c_x - p_c_x) / length)
        return angle, length


