#!/usr/bin/env python3
from geometry_msgs.msg import Pose


def main(context):
    initialP = context.initialPose
    print(initialP)
    print(context.luggagePose)

    p = Pose()
    p.position.x = context.luggagePose[0] - (
        1 * -1 if initialP[0] > context.luggagePose[0] else 1
    )
    p.position.y = context.luggagePose[1] - (
        1 * -1 if initialP[1] > context.luggagePose[1] else 1
    )
    p.position.z = 0
    p.orientation.w = 1
    context.baseController.sync_face_to(
        context.luggagePose[0], context.luggagePose[1]
    )
    context.baseController.sync_to_pose(p)
