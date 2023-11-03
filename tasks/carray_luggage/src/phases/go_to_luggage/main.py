#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose

def main(context):
    # Get robot pose, close distance by 70%
    initialP = context.initialPose
    print(initialP)
    print(context.luggagePose)
    
    #TODO: ADD Safety here
    p = Pose()
    p.position.x = context.luggagePose[0]
    p.position.y = context.luggagePose[1] #- (2 * -1 if initialP[1] > context.luggagePose[1] else 1)
    p.position.z = 0
    p.orientation.w = 1
    context.baseController.sync_to_pose(p)

 
