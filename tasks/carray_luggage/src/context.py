import rospy
from lasr_vision_msgs.srv import YoloDetection
from cv_bridge3 import CvBridge

# from lasr_shapely import LasrShapely
from tf_module.srv import TfTransform
from tiago_controllers.controllers import BaseController, HeadController
from lasr_voice.voice import Voice


class Context:
    def __init__(self):
        self.yolo = rospy.ServiceProxy('/yolov8/detect', YoloDetection)
        rospy.loginfo('YOLO Alive')
        self.bridge = CvBridge()
        rospy.loginfo('CV Bridge Alive')
        # self.shapely = LasrShapely()
        # rospy.loginfo("Shapely Alive")
        self.tf = rospy.ServiceProxy('tf_transform', TfTransform)
        rospy.loginfo('TF Alive')
        self.baseController = BaseController()
        self.headController = HeadController()

        self.initialPose = self.baseController.get_current_pose()
        self.voice = Voice()

        self.luggagePose = None

        rospy.set_param('/is_simulation', False)
