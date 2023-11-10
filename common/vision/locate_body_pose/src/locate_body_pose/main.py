#!/usr/bin/env python3
import rospy
import mediapipe as mp
import cv2
import numpy as np
from locate_body_pose.srv import LocateBodyPose, LocateBodyPoseResponse
from locate_body_pose.settings import POSE_LANDMARK_MODEL
from cv_bridge3 import CvBridge

def draw_landmarks_on_image(rgb_image, detection_result):
  from mediapipe import solutions
  from mediapipe.framework.formats import landmark_pb2

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def getPixel(pose_landmarker_result, img_cv2, idx):
    return int(pose_landmarker_result.pose_landmarks[0][idx].x * len(img_cv2[1])), int(pose_landmarker_result.pose_landmarks[0][idx].y * len(img_cv2))

def main(req):
    res = LocateBodyPoseResponse()
    bridge = CvBridge()
    img_cv2 = bridge.imgmsg_to_cv2_np(req.img)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Put ourselves in the model folder
    import os
    import rospkg
    rp = rospkg.RosPack()
    package_path = rp.get_path('locate_body_pose')
    os.chdir(os.path.abspath(os.path.join(package_path, 'models')))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARK_MODEL),
        running_mode=VisionRunningMode.IMAGE)

    cv2.imwrite("a.png", img_cv2)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_cv2)

    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_img)

        if len(pose_landmarker_result.pose_landmarks) > 0:
            
            annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), pose_landmarker_result)
            cv2.imwrite("b.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            # locs = [13, 15, 14, 16, 11, 12]
            locs = req.points
            pts = []
            vis = []

            for i in locs:
                x, y = getPixel(pose_landmarker_result, img_cv2, i)
                pts.append(x)
                pts.append(y)
                vis.append(pose_landmarker_result.pose_landmarks[0][i].visibility)


            res.cords = pts
            res.vis = vis
            return res


    
    return res



rospy.init_node("locate_body_pose")
rospy.Service("locateBodyPose", LocateBodyPose, main)
rospy.spin()
