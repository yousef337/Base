import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from lasr_object_detection_yolo.detect_objects_v8 import perform_detection, estimate_pose
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsRequest

def verify_expected_env(detections):
    return len(detections) == 3

def analyze_area(context, image_msg, pcl_msg):
    detections, _ = perform_detection(context, pcl_msg, context.polygon, ['person', 'bag'])

    # while not verify_expected_env(detections):
    #     rospy.sleep(2)
    #     detections, _ = perform_detection(context, pcl_msg, context.polygon, ['person', 'bag'])

    return detections

def get_pointing_thumb(context, image_msg, pcl_msg):
    locatePointingThumbRequest = LocateTargetedLuggageCordsRequest()
    locatePointingThumbRequest.img = image_msg
    res = rospy.ServiceProxy('desiredLuggageLocator', LocateTargetedLuggageCords)(locatePointingThumbRequest)

    return estimate_pose(context.tf, pcl_msg, [0, 0, [], [res.x, res.y]])


def calculate_angles(detections, point_thumb_pose):
    bags = list(filter(lambda x: x[0][0] == "bag", detections))

    bag0 = bags[0][1]
    bag1 = bags[1][1]
    person_pose = list(filter(lambda x: x[0][0] == "person", detections))[0][1]

    # get bag1-thumb angle
    
    # get bag2-thumb angle
    # get bag1-person angle
    # get bag2-person angle


def get_pointed_pose(detections, point_thumb_pose):
    return calculate_angles(detections, point_thumb_pose)

def main(context):
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
    image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    
    detections = analyze_area(context, image_msg, pcl_msg)
    point_thumb_pose = get_pointing_thumb(image_msg, pcl_msg)
    return get_pointed_pose(detections, point_thumb_pose)

