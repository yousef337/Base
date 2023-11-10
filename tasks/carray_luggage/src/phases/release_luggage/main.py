#!/usr/bin/env python3
import rospy


def main(context):
    context.voice.speak(
        'I will wait 5 seconds for you to de-attach the luggage'
    )
    rospy.sleep(5)
