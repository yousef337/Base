import mediapipe as mp
import cv2
import numpy as np
from math import acos, pi


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
   


# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# def draw_landmarks_on_image(rgb_image, detection_result):

#   pose_landmarks_list = detection_result.pose_landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected poses to visualize.
#   for idx in range(len(pose_landmarks_list)):
    
#     pose_landmarks = pose_landmarks_list[idx]

#     # Draw the pose landmarks.scoreMajorHand
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     pose_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       pose_landmarks_proto,
#       solutions.pose.POSE_CONNECTIONS,
#       solutions.drawing_styles.get_default_pose_landmarks_style())
#   return annotated_image

def main():
    POSE_LANDMARK_MODEL = "models/pose_landmarker_full.task"

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARK_MODEL),
        running_mode=VisionRunningMode.IMAGE)

    img_msg = cv2.imread('img/a.jpeg') 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_msg)

    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_image)

        # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        # cv2.imwrite("a.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        rightHand = scoreMajorHand(pose_landmarker_result, 23, 11, 13, 15)
        leftHand = scoreMajorHand(pose_landmarker_result, 24, 12, 14, 16)

        print(rightHand)
        print(leftHand)
        majorHand = detectMajor(rightHand, leftHand)
        print("----majorHand")
        print(majorHand)
        
        if majorHand == None:
            return
        
        
        # Get wrist pose in map frame by estimatePose (A) from the yolo frame, change detection frame to only contains the wrist location

        wristMapFrame = None
        

        # (B) CHECK The matching angle between the base-bag_i and wrist-bag_i in

main()


#TODO TOMORROW
# Get (A)
# Get (B)
# Make robot move to the selected bag