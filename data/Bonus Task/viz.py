#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D

category = "detected" # Change this to "detected" to visualize groundtruth
# restart the whole program before changing the category to groundtruth

class ObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.detection_sub = rospy.Subscriber(f"/me5413/{category}", Detection2D, self.callback)

    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data.source_img, "8UC3")

        x = int(data.bbox.center.x)
        y = int(data.bbox.center.y)
        w = int(data.bbox.size_x)
        h = int(data.bbox.size_y)
        cv2.rectangle(cv_image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

def main():
    rospy.init_node('object_detector', anonymous=True)
    od = ObjectDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
