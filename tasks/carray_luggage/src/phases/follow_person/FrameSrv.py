#!/usr/bin/env python3
import rospy
from context import Context
from cv_bridge3 import CvBridge
import cv2
from SortPipeline import sort_pipeline
from sensor_msgs.msg import PointCloud2, Image

def getPeople(context):
    img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message(
        '/xtion/depth_registered/points', PointCloud2
    )
    detections = list(
        filter(
            lambda x: x.name in ['person'],
            context.yolo(
                img_msg, 'yolov8n-seg.pt', 0.7, 0.2
            ).detected_objects,
        )
    )

    return img_msg, pcl_msg, detections


def update_frame():
    context = Context()
    bridge = CvBridge()
    j = 0
    while True:
        img, _, detections = getPeople(context)
        # img_cv2 = bridge.imgmsg_to_cv2_np(img)
        # j += 1
        # cv2.imwrite(f"frame-{j}.png", img_cv2)
        sort_pipeline.update_sort(detections)
        rospy.sleep(0.5)

rospy.init_node("updateFrame")
update_frame()
rospy.spin()
