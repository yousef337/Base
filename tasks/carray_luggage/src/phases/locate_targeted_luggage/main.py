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
import point_cloud_utils as pcu
from sensor_msgs.point_cloud2 import read_points_list
from pcl_segmentation.srv import SegmentView, SegmentViewRequest

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

def ros_to_list(pcl_msg):
    return read_points_list(
                pcl_msg,
                field_names=['x', 'y', 'z'],
                skip_nans=True)

def load_cloud():
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud('file.pcd').points)

def analyze_area(context, rightHandPoses, leftHandPoses):
    lookingDir = 0
    majorRight = abs(rightHandPoses) > abs(leftHandPoses)
    hausdroffLimit = 0.01

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
        rospy.wait_for_service('/pcl_segmentation_server/segment_view')
        clusters = rospy.ServiceProxy('/pcl_segmentation_server/segment_view', SegmentView)(True).clusters
        print("RECEIVED")
        minIdx = (-1, -1)

        b = load_cloud()
        bNormalized = list(map(lambda p: [p.x,   p.y,   p.z], b))

        for j in range(len(clusters)):
            a = ros_to_list(clusters[j])[:]

            maxXA = (max(a, key=lambda x: x.x)).x
            minXA = (min(a, key=lambda x: x.x)).x

            maxYA = (max(a, key=lambda x: x.y)).y
            minYA = (min(a, key=lambda x: x.y)).y

            maxZA = (max(a, key=lambda x: x.z)).z
            minZA = (min(a, key=lambda x: x.z)).z


            maxXB = (max(b, key=lambda x: x.x)).x
            minXB = (min(b, key=lambda x: x.x)).x

            maxYB = (max(b, key=lambda x: x.y)).y
            minYB = (min(b, key=lambda x: x.y)).y

            maxZB = (max(b, key=lambda x: x.z)).z
            minZB = (min(b, key=lambda x: x.z)).z

            aNormalized = list(map(lambda p: [(p.x+minXB-minXA),   (p.y+minYB-minYA),   (p.z+minZB-minZA)], a))

            hausdorff_dist, _, _ = pcu.one_sided_hausdorff_distance(np.array(aNormalized), np.array(bNormalized))
            hausdorff_distb, _, _ = pcu.one_sided_hausdorff_distance(np.array(bNormalized), np.array(aNormalized))
            print(hausdorff_dist, hausdorff_distb)
            print(abs(hausdorff_dist - hausdorff_distb))

            if (minIdx[1] < 0 or minIdx[1] > abs(hausdorff_dist - hausdorff_distb)) and hausdorff_dist - hausdorff_distb < hausdroffLimit:
                minIdx = (j, abs(hausdorff_dist - hausdorff_distb), a, aNormalized)
            print("----")

    results.append(to_map_frame(context, clusters[minIdx[0]], minIdx[2]))
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
        return to_map_frame(
            context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]]
        )
    except:
        if c < 10:
            rospy.sleep(1)
            print(c)
            return estimate_xyz_from_single_point(
                context, pcl_msg, x, y, padding, c + 1
            )
        else:
            raise MemoryError()


def estimate_xyz_from_points(context, pcl_msg, points, c=0):
    try:
        xyz = read_points_list(
            pcl_msg, field_names=['x', 'y', 'z'], skip_nans=True, uvs=points
        )
        xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
        avg_xyz = np.average(xyz, axis=0)
        return to_map_frame(
            context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]]
        )
    except:
        if c < 10:
            rospy.sleep(1)
            return estimate_xyz_from_points(context, pcl_msg, points, c + 1)
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
    res = rospy.ServiceProxy('locateBodyPose', LocateBodyPose)(getHandCords)
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
        res = rospy.ServiceProxy('locateBodyPose', LocateBodyPose)(
            getHandCords
        )
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
        leftPoses,
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
    bagsAngles = []

    for bag in detections:
        bagsAngles.append(
            get_angle_between((np.array(bag) - rightWrist), rightVec)
        )
        bagsAngles.append(
            get_angle_between((np.array(bag) - leftWrist), leftVec)
        )

    print('ANGLES')
    print(bagsAngles)

    return detections[int(bagsAngles.index(min(bagsAngles)) / 2)]


def get_pointed_pose(detections, rightVec, leftVec, rightWrist, leftWrist):
    return calculate_angles(
        detections, rightVec, leftVec, rightWrist, leftWrist
    )


def main(context):
    rospy.sleep(1)
    (
        rightVec,
        leftVec,
        rightWrist,
        leftWrist,
        rightHandPoses,
        leftHandPoses,
    ) = get_hand_vectors(context)

    detections = analyze_area(context, rightHandPoses, leftHandPoses)


    context.luggagePose = get_pointed_pose(
        detections, rightVec, leftVec, rightWrist, leftWrist
    )[3]
