import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, Pose
from sensor_msgs.point_cloud2 import read_points_list
from locate_body_pose.srv import (
    LocateBodyPose,
    LocateBodyPoseRequest,
)


def get_person_location(context):
    image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message(
        '/xtion/depth_registered/points', PointCloud2
    )
    detections = list(
        filter(
            lambda x: x.name in ['person'],
            context.yolo(
                image_msg, 'yolov8n-seg.pt', 0.7, 0.2
            ).detected_objects,
        )
    )

    # TODO: Account for visibility, increase it when no there, move 90 degrees until 360 before fail
    # IF NO DETECTION, MOVE 90 DEGREES TIMES BEFORE EXITING, Add a time out
    # IF VIS too low, move head up down left right to ensure better visibility

    if detections and detections[0]:

        pixels = []

        while len(pixels) == 0:
            print('STUCK IN LOW VISIBILITY')
            getShoulderPose = LocateBodyPoseRequest()
            getShoulderPose.img = rospy.wait_for_message(
                '/xtion/rgb/image_raw', Image
            )
            getShoulderPose.points = [11, 12]

            res = rospy.ServiceProxy('locateBodyPose', LocateBodyPose)(
                getShoulderPose
            )
            pixels = np.array(res.cords).reshape(-1, 2)
            vis = res.vis

        print('===========================res')
        print(pixels)
        print(vis)

        return estimate_xyz_from_single_point(
            context, pcl_msg, pixels[0][0], pixels[0][1]
        )

    return []


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


def main(context):
    last_person_pose = get_person_location(context)
    rospy.sleep(1)
    current_person_pose = get_person_location(context)

    stand_still = 0

    while stand_still < 10:

        if (
            len(last_person_pose) > 0
            and len(current_person_pose) > 0
            and current_person_pose.all() != None
            and np.linalg.norm(current_person_pose - last_person_pose) > 0.2
        ):

            last_person_pose = current_person_pose
            initialP = context.baseController.get_current_pose()
            p = Pose()
            p.position.x = current_person_pose[0]
            p.position.y = current_person_pose[1]
            p.orientation.x = initialP[2].x
            p.orientation.y = initialP[2].y
            p.orientation.z = initialP[2].z
            p.orientation.w = initialP[2].w
            context.baseController.sync_to_pose(p)
            context.baseController.sync_face_to(
                current_person_pose[0], current_person_pose[1]
            )
            stand_still = 0
        else:
            stand_still += 1
            rospy.sleep(0.7)

        rospy.sleep(0.5)
        current_person_pose = get_person_location(context)
        if (
            len(current_person_pose) > 0
            and len(last_person_pose) > 0
            and current_person_pose.all() != None
        ):
            print('====================NORM')
            print(np.linalg.norm(current_person_pose - last_person_pose))
