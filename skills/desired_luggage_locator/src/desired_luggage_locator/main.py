#!/usr/bin/env python3
import rospy
import mediapipe as mp
import cv2
import numpy as np
from math import acos, pi
from desired_luggage_locator.srv import LocateTargetedLuggageCords, LocateTargetedLuggageCordsResponse
from settings import POSE_LANDMARK_MODEL
from cv_bridge3 import CvBridge

def toNPArray(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

def scoreMajorHand(results, hipIdx, shoulderIdx, elbowIdx, wristStartIdx):
    pose_landmarker_result_world = results.pose_world_landmarks[0]
    pose_landmarker_result_pose = results.pose_landmarks[0]

    hip, shoulder = toNPArray(pose_landmarker_result_world[hipIdx]), toNPArray(pose_landmarker_result_world[shoulderIdx])
    elbow, wristStart = toNPArray(pose_landmarker_result_world[elbowIdx]), toNPArray(pose_landmarker_result_world[wristStartIdx])

    hipShoulder = -(hip - shoulder)
    shoulderElbow = -(elbow - shoulder)
    elbowWrist = -(wristStart - elbow)

    #score angle between hipShoulder and shoulderElbow
    theta1 = acos(np.dot(hipShoulder, shoulderElbow)/ (np.linalg.norm(hipShoulder) * np.linalg.norm(shoulderElbow))) * 180/pi

    #score angle between shoulderElbow and elbowWrist
    theta2 = acos(np.dot(shoulderElbow, elbowWrist)/ (np.linalg.norm(shoulderElbow) * np.linalg.norm(elbowWrist))) * 180/pi

    return (theta1, theta2, toNPArray(pose_landmarker_result_pose[wristStartIdx]))


def detectMajor(rightAngles, leftAngles):
    if abs(rightAngles[0] - leftAngles[0]) > 1:
        return rightAngles if rightAngles[0] > leftAngles[0] else leftAngles
   
    if abs(rightAngles[1] - leftAngles[1]) < 1:
        return rightAngles if rightAngles[1] < leftAngles[1] else leftAngles

    return None

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

def main(req):
    res = LocateTargetedLuggageCordsResponse()
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
    package_path = rp.get_path('desired_luggage_locator')
    os.chdir(os.path.abspath(os.path.join(package_path, 'models')))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARK_MODEL),
        running_mode=VisionRunningMode.IMAGE)

    cv2.imwrite("a.png", img_cv2)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_cv2)

    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_img)

        if req.getShoulder and len(pose_landmarker_result.pose_landmarks) > 0:
            res.right = False
            
            annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), pose_landmarker_result)
            cv2.imwrite("b.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            print(pose_landmarker_result.pose_landmarks[0][11])
            print(pose_landmarker_result.pose_landmarks[0][12])

            print(pose_landmarker_result.pose_landmarks[0][12].x*len(img_cv2))
            print(pose_landmarker_result.pose_landmarks[0][12].y*len(img_cv2[1]))

            print(len(img_cv2))
            print(len(img_cv2[1]))
            print((pose_landmarker_result.pose_landmarks[0][11].x+pose_landmarker_result.pose_landmarks[0][12].x)/2)
            print((pose_landmarker_result.pose_landmarks[0][11].y+pose_landmarker_result.pose_landmarks[0][12].y)/2)
            # res.x = int(((pose_landmarker_result.pose_landmarks[0][11].x + pose_landmarker_result.pose_landmarks[0][12].x)/2) * len(img_cv2))
            # res.y = int(((pose_landmarker_result.pose_landmarks[0][11].y + pose_landmarker_result.pose_landmarks[0][12].y)/2) * len(img_cv2[1]))
            res.x = int(pose_landmarker_result.pose_landmarks[0][11].x * len(img_cv2[1]))
            res.y = int(pose_landmarker_result.pose_landmarks[0][11].y * len(img_cv2))
            res.vis = pose_landmarker_result.pose_landmarks[0][11].visibility
            return res

        if len(pose_landmarker_result.pose_landmarks) > 0:
            rightHand = scoreMajorHand(pose_landmarker_result, 23, 11, 13, 15)
            leftHand = scoreMajorHand(pose_landmarker_result, 24, 12, 14, 16)

            majorHand = detectMajor(rightHand, leftHand)

            if majorHand != None:
                res.right = majorHand == rightHand
                res.x = int(majorHand[2][0] * len(img_cv2))
                res.y = int(majorHand[2][1] * len(img_cv2[1]))
                return res
    
    return res



rospy.init_node("desired_luggage_locator")
rospy.Service("desiredLuggageLocator", LocateTargetedLuggageCords, main)
rospy.spin()
