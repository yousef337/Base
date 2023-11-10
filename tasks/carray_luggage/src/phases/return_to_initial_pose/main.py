#!/usr/bin/env python3
from geometry_msgs.msg import Pose


def main(context):
    p = Pose()
    p.position.x = context.initialPose[0]
    p.position.y = context.initialPose[1]
    p.position.z = 0
    p.orientation.w = 1
    print(p)
    context.baseController.sync_to_pose(p)
