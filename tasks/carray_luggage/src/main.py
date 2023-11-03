#!/usr/bin/env python3
import rospy
from phases.locate_targeted_luggage.main import main as LocateDesiredLuggage
from phases.go_to_luggage.main import main as GoToLuggage
from phases.return_to_initial_pose.main import main as ReturnToInitialPose
from phases.pick_up_luggage.main import main as PickUpLuggage
from phases.release_luggage.main import main as ReleaseLuggage
from phases.follow_person.main import main as FollowPerson
from context import Context

def main():
    rospy.init_node("carrayLuggage")
    context = Context()

    #TODO: REPLACE ANGLES WITH ELBOW-WRIST BAGS ANGLES
    # LocateDesiredLuggage(context)
    # GoToLuggage(context)
    # PickUpLuggage(context)
    FollowPerson(context)
    # ReleaseLuggage(context)
    # ReturnToInitialPose(context)


main()