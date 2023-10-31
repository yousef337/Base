#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points_list
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsRequest

def verify_expected_env(detections):
    return len(detections) == 3

def analyze_area(context, image_msg):
    detections = list(filter(lambda x: x.name in ['person', 'bag'], context.yolo(image_msg, "yolov8n-seg.pt", 0.7, 0.2).detected_objects))

    # while not verify_expected_env(detections):
    #     rospy.sleep(2)
    #     detections, _ = perform_detection(context, pcl_msg, context.polygon, ['person', 'bag'])

    return detections

from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

def to_map_frame(context, pcl_msg, pose):
    centroid_xyz = pose
    centroid = PointStamped()
    centroid.point = Point(*centroid_xyz)
    centroid.header = pcl_msg.header
    tf_req = TfTransformRequest()
    tf_req.target_frame = String("map")
    tf_req.point = centroid
    response = context.tf(tf_req)

    return np.array([response.target_point.point.x, response.target_point.point.y, response.target_point.point.z])

def get_pointing_thumb(context, image_msg, pcl_msg):
    locatePointingThumbRequest = LocateTargetedLuggageCordsRequest()
    locatePointingThumbRequest.img = image_msg
    res = rospy.ServiceProxy('desiredLuggageLocator', LocateTargetedLuggageCords)(locatePointingThumbRequest)
    print("res")
    print(res)

    xyz = read_points_list(pcl_msg, skip_nans=True, uvs=[(res.x, res.y)])
    print(xyz)

    return to_map_frame(context, pcl_msg, [xyz[0].x, xyz[0].y, xyz[0].z])


def calculate_angles(detections, point_thumb_pose):
    print("CALCULATING ANGLES")
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

from visualization_msgs.msg import Marker

def main(context):
    # print("In Locate")
    # image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    # print("RECEIVED IMG RAW")
    # pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
    # print("RECEIVED THE MSGS")
    # detections = analyze_area(context, image_msg)
    # print("Finished Analyzing")
    # print(detections)
    # point_thumb_pose = get_pointing_thumb(context, image_msg, pcl_msg)
    # print(point_thumb_pose)

    point_thumb_pose = [3.53854624, 2.46436965, 0.37520515]
    marker_pub = rospy.Publisher.publish("/visualization_marker", Marker)
    marker_msg = Marker()
    marker_msg.header.frame_id = "/map"
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.id = 0
    marker_msg.type = Marker.SPHERE
    marker_msg.action = Marker.ADD
    marker_msg.ns = ""
    marker_msg.pose.position.x = point_thumb_pose[0]
    marker_msg.pose.position.y = point_thumb_pose[1]
    marker_msg.pose.position.z = point_thumb_pose[2]
    marker_msg.pose.orientation.w = 1.0
    marker_msg.scale.x = 0.1
    marker_msg.scale.y = 0.1
    marker_msg.scale.z = 0.1
    marker_msg.color.a = 1.0
    marker_msg.color.r = 0.0
    marker_msg.color.g = 1.0
    marker_msg.color.b = 0.0
    marker_pub.publish(marker_msg)
    #TEST UNTIL HERE
    # return get_pointed_pose(detections, point_thumb_pose)

