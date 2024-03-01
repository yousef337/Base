import numpy as np
from .sort import *

class SortPipeline:
    
    def __init__(self):
        self.data = []
        self.sort = Sort()

    def yolov8_xyseg_to_box(xyseg):
        x1 = -1
        x2 = -1
        y1 = -1
        y2 = -1

        xysegArray = np.array(xyseg).reshape(-1, 2)
        for i in xysegArray:
            x1 = i[0] if i[0] < x1 or x1 == -1 else x1
            x2 = i[0] if i[0] > x2 or x2 == -1 else x2

            y1 = i[1] if i[1] < y1 or y1 == -1 else y1
            y2 = i[1] if i[1] > y2 or y2 == -1 else y2

        return [x1, y1, x2, y2]

    def match_xyseg_xywh(detections, xywh):
        return min(list(map(lambda d:(d.xyseg, np.linalg.norm(np.array(xywh) - np.array(SortPipeline.yolov8_xyseg_to_box(d.xyseg)))), detections)), key=lambda k: k[1])[0]
        

    def update_sort(self, detections):
        # print("================================Updated frame")

        detectionsTracking = np.array(list(map(lambda d: SortPipeline.yolov8_xyseg_to_box(d.xyseg) + [d.confidence], detections)))
        
        if (len(detectionsTracking) == 0):
            detectionsTracking = np.empty((0, 5))

        people_ids = self.sort.update(detectionsTracking)
        self.data = list(map(lambda p: (p[4], p[:4], SortPipeline.match_xyseg_xywh(detections, p[:4])), people_ids))
        # print("self.data")
        # print(len(self.data))

    def get_xyseg_by_id(self, id):
        # print("get_xyseg_by_id")
        # print(self.data)
        for i in self.data:
            if i[0] == id:
                return i[2]
            
        return None 
    
    def get_xywh_by_id(self, id):
        for i in self.data:
            if i[0] == id:
                return i[1]
            
        return None

    def get_ids(self):
        return list(map(lambda d: d[0], self.data))

    def get_data(self):
        return self.data
    
sort_pipeline = SortPipeline()