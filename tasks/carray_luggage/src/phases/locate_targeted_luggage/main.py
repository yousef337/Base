#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points_list
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsRequest
from math import acos
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point



def analyze_area(context, image_msg):
    #TODO: Loop until visibility is high from 0 to 360 degrees, only bags, don't double count, return msgs associated with each element

    return list(filter(lambda x: x.name in ['suitcase'], context.yolo(image_msg, "yolov8n-seg.pt", 0.7, 0.2).detected_objects))


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


def uv_group(uv, padding=2):
    lst = [uv]
    for i in range(-padding, padding):
        for j in range(-padding, padding):
            lst.append((uv[0] + i, uv[0] + j))
    return lst

def estimate_xyz_from_single_point(context, pcl_msg, x, y, padding=2):
    xyz = read_points_list(pcl_msg, field_names=["x", "y", "z"], skip_nans=True, uvs=uv_group((x, y), padding))
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])

def estimate_xyz_from_points(context, pcl_msg, points):
    xyz = read_points_list(pcl_msg, field_names=["x", "y", "z"], skip_nans=True, uvs=points)
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])


def get_hand_vectors(context, image_msg, pcl_msg):
    #TODO: Loop until visibility is high from 0 to 360 degrees, up and down
    getHandCords = LocateTargetedLuggageCordsRequest()
    getHandCords.img = image_msg
    getHandCords.points = [13, 15, 14, 16]
    res = rospy.ServiceProxy('desiredLuggageLocator', LocateTargetedLuggageCords)(getHandCords)
    print("res")
    print(res)
    pixels = np.array(res.cords).reshape(-1, 2)

    rightElbow = estimate_xyz_from_single_point(context, pcl_msg, pixels[0][0], pixels[0][1])
    rightWrist = estimate_xyz_from_single_point(context, pcl_msg, pixels[1][0], pixels[1][1])
    leftElbow = estimate_xyz_from_single_point(context, pcl_msg, pixels[2][0], pixels[2][1])
    leftWrist = estimate_xyz_from_single_point(context, pcl_msg, pixels[3][0], pixels[3][1])


    return (rightWrist - rightElbow), (leftWrist - leftElbow), rightWrist, leftWrist


def get_angle_between(a, b):
    return acos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))

def contour(context, img_msg, pts):
    cv_im =  context.bridge.imgmsg_to_cv2_np(img_msg)
    # Compute mask from contours
    mask = np.zeros(shape=cv_im.shape[:2])
    cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))

    # Extract mask indices from bounding box
    indices = np.argwhere(mask)

    return indices

def calculate_angles(context, img_msg, pcl_msg, detections, rightVec, leftVec, rightWrist, leftWrist):
    bags = list(filter(lambda x: x.name == "suitcase", detections))

    bag0c = contour(context, img_msg,  np.array(bags[0].xyseg).reshape(-1, 2)).tolist()
    bag1c = contour(context, img_msg,  np.array(bags[1].xyseg).reshape(-1, 2)).tolist()
    bag0 = estimate_xyz_from_points(context, pcl_msg, bag0c)
    bag1 = estimate_xyz_from_points(context, pcl_msg, bag1c)

    print("BAGS - Person")
    print(bag0, bag1,  rightVec, leftVec)
    # get bag1-thumb angle
    b0r = get_angle_between((np.array(bag0) - rightWrist), rightVec)
    b0l = get_angle_between((np.array(bag0) - leftWrist), leftVec)
    
    # get bag2-thumb angle
    b1r = get_angle_between((np.array(bag1) - rightWrist), rightVec)
    b1l = get_angle_between((np.array(bag1) - leftWrist), leftVec)

    bagsAngles = [b0r, b0l, b1r, b1l]
    print("ANGLES")
    print(bagsAngles)

    return bag0 if bagsAngles.index(min(bagsAngles)) < 2 else bag1


def get_pointed_pose(context, img_msg, pcl_msg, detections, rightVec, leftVec, rightWrist, leftWrist):
    return calculate_angles(context, img_msg, pcl_msg, detections, rightVec, leftVec, rightWrist, leftWrist)


def main(context):
    image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)

    detections = analyze_area(context, image_msg)
    rightVec, leftVec, rightWrist, leftWrist = get_hand_vectors(context, image_msg, pcl_msg)

    context.luggagePose = get_pointed_pose(context, image_msg, pcl_msg, detections, rightVec, leftVec, rightWrist, leftWrist)

