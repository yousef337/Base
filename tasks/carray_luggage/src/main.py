#!/usr/bin/env python3
import rospy
from phases.locate_targeted_luggage.main import main as LocateDesiredLuggage
from context import Context

def main():
    rospy.init_node("carrayLuggage")
    context = Context()

    luggage_pose = LocateDesiredLuggage(context)
    print(luggage_pose)
    # GO TO IT, AKA Phase 2


main()