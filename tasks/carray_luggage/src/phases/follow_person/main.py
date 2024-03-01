import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, Pose
from sensor_msgs.point_cloud2 import read_points_list
from .sort import *
from locate_body_pose.srv import (
    LocateBodyPose,
    LocateBodyPoseRequest,
)
from cv_bridge3 import CvBridge
from .SortPipeline import sort_pipeline

trackingId = None
j = 0
k = 0
d = 0

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


def getHandData(img):
    getHandCords = LocateBodyPoseRequest()
    getHandCords.img = img
    getHandCords.points = [0]
    resPose = rospy.ServiceProxy('locateBodyPose', LocateBodyPose)(getHandCords).pose

    return resPose[0].cords if len(resPose) > 0 else None

def cropPipeline(img, sort_pipeline):
    bridge = CvBridge()
    img_cv2 = bridge.imgmsg_to_cv2_np(img)
    poses = []

    for i in sort_pipeline.get_ids():
        xywh = sort_pipeline.get_xywh_by_id(i)
        # cv2.imwrite("im.png", img_cv2[int(xywh[1]):int(xywh[3]), int(xywh[0]):int(xywh[2])])
        croppedImgMsg = bridge.cv2_to_imgmsg(img_cv2[int(xywh[1]):int(xywh[3]), int(xywh[0]):int(xywh[2])])
        poses.append((getHandData(croppedImgMsg), i))

    return poses


def getRaisingHandPerson(img_msg_before, img_msg_after, sort_pipeline):
    before = cropPipeline(img_msg_before, sort_pipeline)
    after = cropPipeline(img_msg_after, sort_pipeline)
    change = []

    # print("BEFORE & AFTER")
    # print(before)
    # print(after)

    for i in range(len(after)):
        for j in range(len(before)):
            if after[i][1] == before[j][1]:
                if (after[i][0] is not None and before[j][0] is not None):
                    change.append((after[i][1], np.linalg.norm(np.array(after[i][0]) - np.array(before[j][0]))))
                else:
                    # print(after[i][0])
                    # print(before[j][0])
                    change.append((after[i][1], 0))
    
    
    return max(change, key=lambda c: c[1])[0] if len(change) > 0 else None

def adjustHead(context, person_frame, image_frame):
    global d
    global j
    cv2.imwrite(f"followJ-{j}.png", image_frame)
    center = np.array(person_frame[0] + (person_frame[2]/2))
    print("-------------center")
    print(person_frame)
    print(center)
    if center > 500:
        d -= 0.2
        print("-------------DIR A")
        context.headController.look_xy(d, 0)

    elif center < 150:
        d += 0.2
        print("-------------DIR B")
        context.headController.look_xy(d, 0)

def get_person_location(context):
    global trackingId
    global j
    global k
    global d

    looks = [
        # None,
        context.headController.look_straight,
        context.headController.look_left,
        context.headController.look_right,
    ]

    for i in looks:

        if trackingId == None:
            rospy.sleep(0.5)
            i()

        img_msg_before, pcl_msg, detections = getPeople(context)
        last_img_msg = img_msg_before
        sort_pipeline.update_sort(detections)


        if detections and detections[0]:
            if trackingId == None:
                context.voice.speak("Raise your hand if you want me to follow you.")        
                rospy.sleep(1)
                img_msg_after, pcl_msg, detections = getPeople(context)
                last_img_msg = img_msg_after
                sort_pipeline.update_sort(detections)

                trackingId = getRaisingHandPerson(img_msg_before, img_msg_after, sort_pipeline)

                bridge = CvBridge()
                img_cv2 = bridge.imgmsg_to_cv2_np(last_img_msg)
                j += 1
                # cv2.imwrite(f"follow-{j}.png", img_cv2)
                if trackingId == None:

                    if k == 100:
                        context.voice.speak("I can't see the tracked person")
                        k = 0
                    k += 1 
                    return []

                personXyseg = sort_pipeline.get_xyseg_by_id(trackingId)
                context.voice.speak("I got you.")
                # print(trackingId)

            else:
                # print("-----------_HERE 111")
                personXyseg = sort_pipeline.get_xyseg_by_id(trackingId)
                
                if personXyseg is None:

                    if k == 100:
                        context.voice.speak("I can't see the tracked person")
                        trackingId = None
                        k = 0
                    k += 1 
                    
                    bridge = CvBridge()
                    img_cv2 = bridge.imgmsg_to_cv2_np(last_img_msg)
                    j += 1
                    cv2.imwrite(f"follow-{j}.png", img_cv2)

                    return []

            bridge = CvBridge()
            img_cv2 = bridge.imgmsg_to_cv2_np(last_img_msg)
            
            j += 1
            cv2.imwrite(f"follow-{j}.png", img_cv2)

            adjustHead(context, sort_pipeline.get_xywh_by_id(trackingId), img_cv2)
            k = 0
            return estimate_xyz_from_points(
                context, pcl_msg, contour(context, last_img_msg, np.array(personXyseg).reshape(-1, 2)).tolist()
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

    # cv2.imwrite("a.png", mask)

    return indices

def estimate_xyz_from_points(context, pcl_msg, points, c=0):
    try:
        xyz = read_points_list(
            pcl_msg, field_names=['x', 'y', 'z'], skip_nans=True, uvs=map(lambda x: (x[1], x[0]), points)
        )
        xyz = list(map(lambda x: [x.x, x.y, x.z], xyz))
        avg_xyz = np.average(xyz, axis=0)
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


def marker(a):
    from visualization_msgs.msg import Marker
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=2)
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "base_footprint"
    marker.type = 8
    marker.type = Marker.SPHERE

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.pose.position.x = a.x
    marker.pose.position.y = a.y
    marker.pose.position.z = 0
    marker.pose.orientation.w = 1
    i = 0
    while i < 2:
        marker_pub.publish(marker)
        rospy.sleep(1)
        i += 1

def mainmain(context):
    last_person_pose = get_person_location(context)[:2]
    robot_pose = context.baseController.get_current_pose()
    stand_still = 0
    
    
    while stand_still < 50:
        # print(last_person_pose)
        # print([robot_pose[0], robot_pose[1]])
        if (
            len(last_person_pose) > 0
            and np.linalg.norm(np.array([robot_pose[0], robot_pose[1]]) - last_person_pose) > 1
        ):

            ori = context.baseController.compute_face_quat(last_person_pose[0], last_person_pose[1]).orientation
            p = Pose()
            p.position.x = last_person_pose[0]
            # p.position.x = robot_pose[0]
            p.position.y = last_person_pose[1]
            # p.position.y = robot_pose[1]
            p.orientation = ori
            # context.baseController.async_face_to(last_person_pose[0], last_person_pose[1])

            if abs(last_person_pose[0] - robot_pose[0]) > 1:
                p.position.x -= 1 if last_person_pose[0] > robot_pose[0] else -1
            
            if abs(last_person_pose[1] - robot_pose[1]) > 1:
                p.position.y -= 1 if last_person_pose[1] > robot_pose[1] else -1
            
            marker(p.position)
            # print('====================POSE')
            # print(p.position)
            context.baseController.async_to_pose(p)
            robot_pose = context.baseController.get_current_pose()

            stand_still = 0
        else:
            stand_still += 1

        # rospy.sleep(1)
        last_person_pose = get_person_location(context)[:2]
        # if (len(last_person_pose) > 0):
        #     print('====================NORM')
        #     print(np.linalg.norm(np.array([robot_pose[0], robot_pose[1]]) - last_person_pose))


def main(context):
    mainmain(context)
   
    

