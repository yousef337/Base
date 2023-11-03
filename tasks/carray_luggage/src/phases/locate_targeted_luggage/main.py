#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points_list
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsRequest
from math import acos
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

def verify_expected_env(detections):
    return len(detections) == 3

def analyze_area(context, image_msg):
    detections = list(filter(lambda x: x.name in ['person', 'bag', 'suitcase'], context.yolo(image_msg, "yolov8n-seg.pt", 0.7, 0.2).detected_objects))

    # while not verify_expected_env(detections):
    #     rospy.sleep(2)
    #     detections, _ = perform_detection(context, pcl_msg, context.polygon, ['person', 'bag'])

    return detections


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


def uv_group(uv, padding=10):
    lst = [uv]
    for i in range(-padding, padding):
        for j in range(-padding, padding):
            lst.append((uv[0] + i, uv[0] + j))
    return lst

def estimate_xyz_from_single_point(context, pcl_msg, x, y, padding=10):
    xyz = read_points_list(pcl_msg, field_names=["x", "y", "z"], skip_nans=True, uvs=uv_group((x, y), padding))
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])

def estimate_xyz_from_points(context, pcl_msg, points):
    xyz = read_points_list(pcl_msg, field_names=["x", "y", "z"], skip_nans=True, uvs=points)
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])


def get_pointing_thumb(context, image_msg, pcl_msg):
    locatePointingThumbRequest = LocateTargetedLuggageCordsRequest()
    locatePointingThumbRequest.img = image_msg
    res = rospy.ServiceProxy('desiredLuggageLocator', LocateTargetedLuggageCords)(locatePointingThumbRequest)
    print("res")
    print(res)

    return estimate_xyz_from_single_point(context, pcl_msg, res.x, res.y)


def get_angle_between(a, b):
    return acos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))

def draw(luggagePose, id):
    from visualization_msgs.msg import Marker

    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 12)

    marker = Marker()

    marker.header.frame_id = "/map"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 0
    marker.id = id
    marker.ns = "p"
    marker.action = 0

    # Set the scale of the marker
    marker.scale.x = 3.0
    marker.scale.y = 3.0
    marker.scale.z = 3.0

    # Set the color
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    marker.pose.position.x =  luggagePose[0]
    marker.pose.position.y = luggagePose[1]
    marker.pose.position.z = luggagePose[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.lifetime = 0
    marker_pub.publish(marker)

def calculate_angles(context, pcl_msg, detections, point_thumb_pose):
    bags = list(filter(lambda x: x.name == "suitcase", detections))
    people = list(filter(lambda x: x.name == "person", detections))

    bag0 = estimate_xyz_from_points(context, pcl_msg, np.array(bags[0].xyseg).reshape(-1, 2).tolist())
    draw(bag0, 0)
    bag1 = estimate_xyz_from_points(context, pcl_msg, np.array(bags[1].xyseg).reshape(-1, 2).tolist())
    draw(bag1, 1)
    person_pose = estimate_xyz_from_points(context, pcl_msg, np.array(people[0].xyseg).reshape(-1, 2).tolist())
    draw(person_pose, 2)
    print("BAGS - Person")
    print(bag0, bag1, person_pose)
    # get bag1-thumb angle
    b0t = get_angle_between(bag0, point_thumb_pose)
    
    # get bag2-thumb angle
    b1t = get_angle_between(bag1, point_thumb_pose)

    # get bag1-person angle
    b0p = get_angle_between(bag0, person_pose)

    # get bag2-person angle
    b1p = get_angle_between(bag1, person_pose)

    # print(b0t, b1t, b0p, b1p)
    b0tpDiff = abs(b0t-b0p)
    b1tpDiff = abs(b1t-b1p)
    # print(b0tpDiff, b1tpDiff)

    return bag0 if b0tpDiff < b1tpDiff else bag1


def get_pointed_pose(context, pcl_msg, detections, point_thumb_pose):
    return calculate_angles(context, pcl_msg, detections, point_thumb_pose)


def main(context):
    image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)

    detections = analyze_area(context, image_msg)
    point_thumb_pose = get_pointing_thumb(context, image_msg, pcl_msg)

    context.luggagePose = get_pointed_pose(context, pcl_msg, detections, point_thumb_pose)

