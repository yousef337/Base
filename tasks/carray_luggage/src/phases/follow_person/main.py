import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, Pose
from sensor_msgs.point_cloud2 import read_points_list


def get_person_location(context):

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
                lambda x: x.name in ['person'],
                context.yolo(
                    img_msg, 'yolov8n-seg.pt', 0.7, 0.2
                ).detected_objects,
            )
        )

        if detections and detections[0]:
            return estimate_xyz_from_points(
                context, pcl_msg, contour(context, img_msg, np.array(detections[0].xyseg).reshape(-1, 2)).tolist()
        )

    return []

def contour(context, img_msg, pts):
    cv_im = context.bridge.imgmsg_to_cv2_np(img_msg)
    # Compute mask from contours
    mask = np.zeros(shape=cv_im.shape[:2])
    cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))

    # Extract mask indices from bounding box
    indices = np.argwhere(mask)

    # SAVE

    cv2.imwrite("a.png", mask)

    return indices

def estimate_xyz_from_points(context, pcl_msg, points, c=0):
    try:
        xyz = read_points_list(
            pcl_msg, field_names=['x', 'y', 'z'], skip_nans=True, uvs=map(lambda x: (x[1], x[0]), points)
        )
        xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
        avg_xyz = np.average(xyz, axis=0)
        print("np.average(points, axis=0)")
        print(np.average(points, axis=0))
        return to_map_frame(
            context, pcl_msg, [avg_xyz[0], avg_xyz[1], avg_xyz[2]]
        )
    except:
        if c < 10:
            print(c)
            rospy.sleep(1)
            return estimate_xyz_from_points(context, pcl_msg, points, c + 1)
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
    last_person_pose = get_person_location(context)[:2]
    robot_pose = context.baseController.get_current_pose()
    print(last_person_pose)
    print(robot_pose[:2])
    stand_still = 0
    
    while stand_still < 10:

        if (
            len(last_person_pose) > 0
            and np.linalg.norm(np.array([robot_pose[0], robot_pose[1]]) - last_person_pose) > 1.5
        ):

            print('====================NORM')
            print(np.linalg.norm(np.array([robot_pose[0], robot_pose[1]]) - last_person_pose))


            p = Pose()
            p.position.x = last_person_pose[0]
            p.position.y = last_person_pose[1]
            p.orientation.x = robot_pose[2].x
            p.orientation.y = robot_pose[2].y
            p.orientation.z = robot_pose[2].z
            p.orientation.w = robot_pose[2].w

            if abs(last_person_pose[0] - robot_pose[0]) > 2:
                p.position.x -= 1 if last_person_pose[0] > robot_pose[0] else -1
            
            if abs(last_person_pose[1] - robot_pose[1]) > 2:
                p.position.y -= 1 if last_person_pose[1] > robot_pose[1] else -1
            
            context.baseController.sync_to_pose(p)
            context.baseController.sync_to_pose(p)

            stand_still = 0
        else:
            stand_still += 1
            rospy.sleep(0.7)

        rospy.sleep(0.5)
        last_person_pose = get_person_location(context)[:2]
        robot_pose = context.baseController.get_current_pose()
        if (len(last_person_pose) > 0):
            print('====================NORM')
            print(np.linalg.norm(np.array([robot_pose[0], robot_pose[1]]) - last_person_pose))
