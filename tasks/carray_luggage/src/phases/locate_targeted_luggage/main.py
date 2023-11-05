#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points_list
from desired_luggage_locator.srv import (
    LocateTargetedLuggageCords,
    LocateTargetedLuggageCordsRequest,
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


def analyze_area(context):
    results = []
    looks = [
        context.headController.look_straight,
        context.headController.look_left,
        context.headController.look_right,
    ]

    for i in looks:
        i()
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        pcl_msg = rospy.wait_for_message(
            '/xtion/depth_registered/points', PointCloud2
        )
        detections = list(
            filter(
                lambda x: x.name in ['suitcase', 'person'],
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


def estimate_xyz_from_single_point(context, pcl_msg, x, y, padding=1):
    xyz = read_points_list(
        pcl_msg,
        field_names=['x', 'y', 'z'],
        skip_nans=True,
        uvs=uv_group((x, y), padding),
    )
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])


def estimate_xyz_from_points(context, pcl_msg, points):
    xyz = read_points_list(
        pcl_msg, field_names=['x', 'y', 'z'], skip_nans=True, uvs=points
    )
    xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
    avg_xyz = np.average(xyz, axis=0)
    return to_map_frame(context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]])


def get_hand_vectors(context, detection):

    context.baseController.sync_face_to(detection[3][0], detection[3][1])
    getHandCords = LocateTargetedLuggageCordsRequest()
    getHandCords.img = detection[2]
    getHandCords.points = [13, 15, 14, 16]
    res = rospy.ServiceProxy(
        'desiredLuggageLocator', LocateTargetedLuggageCords
    )(getHandCords)
    print('res')
    print(res)
    pixels = np.array(res.cords).reshape(-1, 2)

    rightElbow = estimate_xyz_from_single_point(
        context, detection[1], pixels[0][0], pixels[0][1]
    )
    rightWrist = estimate_xyz_from_single_point(
        context, detection[1], pixels[1][0], pixels[1][1]
    )
    leftElbow = estimate_xyz_from_single_point(
        context, detection[1], pixels[2][0], pixels[2][1]
    )
    leftWrist = estimate_xyz_from_single_point(
        context, detection[1], pixels[3][0], pixels[3][1]
    )

    return (
        (rightWrist - rightElbow),
        (leftWrist - leftElbow),
        rightWrist,
        leftWrist,
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
    detections = analyze_area(context)
    rightVec, leftVec, rightWrist, leftWrist = get_hand_vectors(
        context,
        detections[(list(map(lambda d: d[0], detections))).index('person')],
    )
    context.luggagePose = get_pointed_pose(
        detections, rightVec, leftVec, rightWrist, leftWrist
    )[3]
