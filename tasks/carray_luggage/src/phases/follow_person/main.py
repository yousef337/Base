import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, Image
from tf_module.srv import TfTransformRequest
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, Pose
from sensor_msgs.point_cloud2 import read_points_list
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsRequest

def get_person_location(context):
    image_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
    detections = list(filter(lambda x: x.name in ['person'], context.yolo(image_msg, "yolov8n-seg.pt", 0.7, 0.2).detected_objects))

    # TODO: Account for visibility, increase it when no there, move 90 degrees until 360 before fail

    if detections and detections[0]:

        getShoulderPose = LocateTargetedLuggageCordsRequest()
        getShoulderPose.img = image_msg
        getShoulderPose.points = [11, 12]

        res = rospy.ServiceProxy('desiredLuggageLocator', LocateTargetedLuggageCords)(getShoulderPose)
        pixels = np.array(res.cords).reshape(-1, 2)

        return estimate_xyz_from_single_point(context, pcl_msg, int((pixels[0][0]+pixels[1][0])/2), int((pixels[0][1]+pixels[1][0])/2))
    
    return None

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


def main(context):
    last_person_pose = get_person_location(context)
    print("TAKEN")
    rospy.sleep(1)
    current_person_pose = get_person_location(context)

    print("====================NORM")
    print(np.linalg.norm(current_person_pose - last_person_pose))

    while current_person_pose.all() != None and np.linalg.norm(current_person_pose - last_person_pose) > 0.5:
        last_person_pose = current_person_pose
        initialP = context.baseController.get_current_pose()
        #TODO: fix orientation to whereever orientation is human -> a function in controllers
        #TODO: fix exiting condition to be run over 10 times
        p = Pose()
        p.position.x = current_person_pose[0] #- (1.5 * -1 if initialP[0] > current_person_pose[0] else 1)
        p.position.y = current_person_pose[1] #- (1.5 * -1 if initialP[1] > current_person_pose[1] else 1)
        p.position.z = 0
        p.orientation.x = initialP[2].x
        p.orientation.y = initialP[2].y
        p.orientation.z = initialP[2].z
        p.orientation.w = initialP[2].w
        context.baseController.sync_to_pose(p)
        
        rospy.sleep(0.5)
        current_person_pose = get_person_location(context)
        print("====================NORM")
        print(np.linalg.norm(current_person_pose - last_person_pose))
