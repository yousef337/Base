#!/usr/bin/env python3
import rospy


def main(context):
    context.voice('I will wait 5 seconds for you to attach the luggage')
    rospy.sleep(5)
