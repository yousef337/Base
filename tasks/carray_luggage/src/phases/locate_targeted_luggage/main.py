#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points_list
from locate_body_pose.srv import (
    LocateBodyPose,
    LocateBodyPoseRequest,
)
from math import acos
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point


def in_result(results, entry):
    return (
        len(results) > 0
        and entry[0] in list(map(lambda r: r[0], results))
        and min(
            list(
                map(
                    lambda r: np.linalg.norm(
                        np.array(r[3]) - np.array(entry[1])
                    ),
                    results,
                )
            )
        )
        < 0.1
    )


def analyze_area(context, rightHandPoses, leftHandPoses):
    lookingDir = 0
    majorRight = abs(rightHandPoses) > abs(leftHandPoses)

    if majorRight and rightHandPoses < 0:
        lookingDir = 2
    elif majorRight and rightHandPoses > 0:
        lookingDir = 1
    elif not majorRight and leftHandPoses > 0:
        lookingDir = 1
    elif not majorRight and leftHandPoses < 0:
        lookingDir = 2

    results = []
    looks = [
        context.headController.look_straight_down,
        context.headController.look_left_down,
        context.headController.look_right_down,
    ]

    for i in [looks[0], looks[lookingDir]]:
        i()
        rospy.sleep(1)
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        pcl_msg = rospy.wait_for_message(
            '/xtion/depth_registered/points', PointCloud2
        )
        detections = list(
            filter(
                lambda x: x.name in ['suitcase'],
                context.yolo(
                    img_msg, 'yolov8n-seg.pt', 0.7, 0.2
                ).detected_objects,
            )
        )
        mappedDetections = list(
            map(
                lambda d: (
                    d.name,
                    estimate_xyz_from_points(
                        context,
                        pcl_msg,
                        contour(
                            context, img_msg, np.array(d.xyseg).reshape(-1, 2)
                        ).tolist(),
                    ),
                ),
                detections,
            )
        )

        for entry in mappedDetections:
            if not in_result(results, entry):
                results.append((entry[0], pcl_msg, img_msg, entry[1]))

    return results


def to_map_frame(context, pcl_msg, pose):
    centroid_xyz = pose
    centroid = PointStamped()
    centroid.point = Point(*centroid_xyz)
    centroid.header = pcl_msg.header
    tf_req = TfTransformRequest()
    tf_req.target_frame = String('map')
    tf_req.point = centroid
    response = context.tf(tf_req)

    return np.array(
        [
            response.target_point.point.x,
            response.target_point.point.y,
            response.target_point.point.z,
        ]
    )


def uv_group(uv, padding=1):
    lst = [uv]
    for i in range(-padding, padding):
        for j in range(-padding, padding):
            lst.append((uv[0] + i, uv[0] + j))
    return lst


def estimate_xyz_from_single_point(context, pcl_msg, x, y, padding=2, c=0):
    try:
        xyz = read_points_list(
            pcl_msg,
            field_names=['x', 'y', 'z'],
            skip_nans=True,
            uvs=uv_group((x, y), padding),
        )
        xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
        avg_xyz = np.average(xyz, axis=0)
        return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])
    except:
        if c < 10:
            rospy.sleep(1)
            print(c)
            return estimate_xyz_from_single_point(context, pcl_msg, x, y, padding, c+1)
        else:
            raise MemoryError()

def estimate_xyz_from_points(context, pcl_msg, points, c=0):
    try:
        xyz = read_points_list(
            pcl_msg, field_names=['x', 'y', 'z'], skip_nans=True, uvs=points
        )
        xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
        avg_xyz = np.average(xyz, axis=0)
        return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])
    except:
        if c < 10:
            rospy.sleep(1)
            return estimate_xyz_from_points(context, pcl_msg, points, c+1)
        else:
            raise MemoryError()

def get_hand_vectors(context):
    context.headController.look_straight()
    img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message(
        '/xtion/depth_registered/points', PointCloud2
    )
    
    getHandCords = LocateBodyPoseRequest()
    getHandCords.img = img_msg
    getHandCords.points = [13, 15, 14, 16]
    res = rospy.ServiceProxy(
        'locateBodyPose', LocateBodyPose
    )(getHandCords)
    pixels = np.array(res.cords).reshape(-1, 2)
    vis = res.vis

    c = 0
    while (len(pixels) < 1 or sum(vis) < 2) and c < 10:
        rospy.sleep(1)
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        pcl_msg = rospy.wait_for_message(
            '/xtion/depth_registered/points', PointCloud2
        )

        getHandCords.img = img_msg
        res = rospy.ServiceProxy(
            'locateBodyPose', LocateBodyPose
        )(getHandCords)
        pixels = np.array(res.cords).reshape(-1, 2)
        vis = res.vis
        print(pixels)
        print(vis)
        c += 1

    print('res')
    print(res)

    rightElbow = estimate_xyz_from_single_point(
        context, pcl_msg, pixels[0][0], pixels[0][1]
    )
    rightWrist = estimate_xyz_from_single_point(
        context, pcl_msg, pixels[1][0], pixels[1][1]
    )
    leftElbow = estimate_xyz_from_single_point(
        context, pcl_msg, pixels[2][0], pixels[2][1]
    )
    leftWrist = estimate_xyz_from_single_point(
        context, pcl_msg, pixels[3][0], pixels[3][1]
    )

    rightPoses = pixels[0][0] - pixels[1][0]

    leftPoses = pixels[2][0] - pixels[3][0]

    return (
        -(rightWrist - rightElbow),
        -(leftWrist - leftElbow),
        rightWrist,
        leftWrist,
        rightPoses,
        leftPoses
    )


def get_angle_between(a, b):
    return acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def contour(context, img_msg, pts):
    cv_im = context.bridge.imgmsg_to_cv2_np(img_msg)
    # Compute mask from contours
    mask = np.zeros(shape=cv_im.shape[:2])
    cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))

    # Extract mask indices from bounding box
    indices = np.argwhere(mask)

    return indices


def calculate_angles(detections, rightVec, leftVec, rightWrist, leftWrist):
    bags = list(filter(lambda x: x[0] == 'suitcase', detections))
    bagsAngles = []

    for bag in bags:
        bagsAngles.append(
            get_angle_between((np.array(bag[3]) - rightWrist), rightVec)
        )
        bagsAngles.append(
            get_angle_between((np.array(bag[3]) - leftWrist), leftVec)
        )

    print('ANGLES')
    print(bagsAngles)

    return bags[int(bagsAngles.index(min(bagsAngles)) / 2)]


def get_pointed_pose(detections, rightVec, leftVec, rightWrist, leftWrist):
    return calculate_angles(
        detections, rightVec, leftVec, rightWrist, leftWrist
    )


def main(context):
    rospy.sleep(1)
    rightVec, leftVec, rightWrist, leftWrist, rightHandPoses, leftHandPoses = get_hand_vectors(context)
    print("==================")
    print(rightHandPoses)
    print(leftHandPoses)
    detections = analyze_area(context, rightHandPoses, leftHandPoses)
    context.luggagePose = get_pointed_pose(
        detections, rightVec, leftVec, rightWrist, leftWrist
    )[3]
