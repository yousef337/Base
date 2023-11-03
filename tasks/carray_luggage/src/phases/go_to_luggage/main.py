#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose

def main(context):
    # Get robot pose, close distance by 70%
    initialP = context.initialPose
    print(initialP)
    print(context.luggagePose)
    
    p = Pose()
    p.position.x = context.luggagePose[0]  - (1.5 * -1 if initialP[0] > context.luggagePose[0] else 1)
    p.position.y = context.luggagePose[1]
    p.position.z = context.luggagePose[2]
    p.orientation.w = 1
    context.baseController.sync_to_pose(p)

 
