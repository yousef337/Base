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


def main(req):
    res = LocateTargetedLuggageCordsResponse()
    bridge = CvBridge()
    img_cv2 = bridge.imgmsg_to_cv2_np(req.img)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARK_MODEL),
        running_mode=VisionRunningMode.IMAGE)


    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(img_cv2)
        
        rightHand = scoreMajorHand(pose_landmarker_result, 23, 11, 13, 15)
        leftHand = scoreMajorHand(pose_landmarker_result, 24, 12, 14, 16)

        majorHand = detectMajor(rightHand, leftHand)

        if majorHand != None:
            res.x = int(majorHand[2][0] * len(img_cv2))
            res.y = int(majorHand[2][1] * len(img_cv2[1]))
            return res
    
    return res



rospy.init_node("desired_luggage_locator")
rospy.Service("desiredLuggageLocator", LocateTargetedLuggageCords)
rospy.spin()
