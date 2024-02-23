#!usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from std_msgs.msg import String
from vision_msgs.msg import Detection2D

seq_num = 'seq_1'

groundtruth = []
with open(f'src/beginner_tutorials/data/{seq_num}/groundtruth.txt', 'r') as file:
    for line in file:
        groundtruth.append(list(map(int, line.strip().split(','))))

firsttrack_list = []
with open(f'src/beginner_tutorials/data/{seq_num}/firsttrack.txt', 'r') as file:
    x, y, w, h = map(int, file.readline().strip().split(','))
firsttrack_list.append((x, y, w, h))
    
def crop_result(frame, match_x, match_y, match_width, match_hight):
    # Preprocess the matched region
    matched_region = frame[match_y:match_y+match_hight,match_x:match_x+match_width]
    matched_region_gray = cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(matched_region_gray, (7, 7), 0)
    equalized = cv2.equalizeHist(blurred)
    # Edge detection
    edge_detected_image = cv2.Canny(equalized, 100, 200)
    contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, update the size based on the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_b, y_b, w_b, h_b = cv2.boundingRect(largest_contour)
        # Cropped region can't be smaller than 75% of the original area
        original_area = match_hight * match_width
        min_area_threshold = original_area * 0.75
        if w_b * h_b < min_area_threshold:
            return match_x, match_y, match_width, match_hight
        x_c, y_c =match_x + x_b, match_y + y_b
        w_c, h_c = w_b, h_b
    else:
        x_c, y_c = match_x, match_y
        w_c, h_c = match_width, match_hight 

    return x_c, y_c, w_c, h_c

# Calculate the direction vector
def calculate_direction_vector(prev_prev, prev):
    center_prev_prev = (prev_prev[0] + prev_prev[2] / 2, prev_prev[1] + prev_prev[3] / 2)
    center_prev = (prev[0] + prev[2] / 2, prev[1] + prev[3] / 2)
    direction_vector = (center_prev[0] - center_prev_prev[0], center_prev[1] - center_prev_prev[1])
    return direction_vector

# Timing-based dynamic template match
def timing_base_dynamic_template_match(frame, template, prev, prev_prev, search_padding=10, direction_scale=0.8):
    x_prev, y_prev, w_prev, h_prev = prev

    # Calculate the direction vector and its module
    direction_vector = calculate_direction_vector(prev_prev, prev)
    module = np.linalg.norm(direction_vector)
    # Generate the dynamic padding and direction scale
    dynamic_padding = int(search_padding + module)
    dynamic_direction_scale = direction_scale
    if module < 10:
        dynamic_direction_scale = 0.4
    
    # According to the direction vector, calculate the start point of the search window
    search_x_start = max(0, int(x_prev + direction_vector[0]*dynamic_direction_scale))
    search_y_start = max(0, int(y_prev + direction_vector[1]*dynamic_direction_scale))
    
    # Define the size of the search window
    search_w = int(w_prev + dynamic_padding)
    search_h = int(h_prev + dynamic_padding)
    # Ensure the search window is within the frame
    frame_height, frame_width = frame.shape[:2]
    search_w = min(search_w, frame_width - search_x_start)
    search_h = min(search_h, frame_height - search_y_start)
    
    # Adjust the start point of the search window
    search_x = max(0, search_x_start + w_prev // 2 - search_w // 2)
    search_y = max(0, search_y_start + h_prev // 2 - search_h // 2)
    # Ensure the search window is within the frame
    # Or return the previous result
    search_region = frame[search_y:search_y + search_h, search_x:search_x + search_w]
    if search_w < template.shape[1] or search_h < template.shape[0]:
        return prev
    
    # Template match
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the matched region
    match_width, match_hight = int(template.shape[1]), int(template.shape[0])
    match_x, match_y = max_loc[0] + search_x, max_loc[1] + search_y
    # Crop the matched region based on the contours
    x_c, y_c, w_c, h_c = crop_result(frame, match_x, match_y, match_width, match_hight)
    x_d, y_d, w_d, h_d = x_c, y_c, w_c, h_c
    return x_d, y_d, w_d, h_d

class Detector(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/me5413/image_raw", Image, self.callback)
        # Template for matching, obtained from the first frame.
        self.template = None
        self.template_coords = firsttrack_list[0]  # Initial tracking coordinates (x, y, w, h)
        self.prev = None
        self.prev_prev = None

        # queue init
        self.frame_queue = Queue()
        self.stop_thread = False
        self.process_thread = Thread(target=self.process_frames)
        self.process_thread.start()

        # Initialize a publisher for sending msg
        self.matrix_pub = rospy.Publisher("/me5413/student_matrix", String, queue_size=10)
        self.detected_pub = rospy.Publisher("/me5413/detected", Detection2D, queue_size=10)
        self.groundtruth_pub = rospy.Publisher("/me5413/groundtruth", Detection2D, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
            self.frame_queue.put(cv_image)
        except CvBridgeError as e:
            print(e)

    def process_frames(self):
        frame_number = 0  # ground truth index
        while not self.stop_thread:
            if not self.frame_queue.empty():
                cv_image = self.frame_queue.get()
                if self.template is None:
                    # Initialize template with the first frame using initial coordinates.
                    x, y, w, h = self.template_coords
                    self.template = cv_image[y:y+h, x:x+w]
                    self.prev = self.template_coords
                    self.prev_prev = self.prev
                else:
                    frame = self.detect(cv_image, frame_number)
                    frame_number += 1 # index update
                # Publish the string message
                self.matrix_pub.publish("A0285282X") 

    def publish_detection(self, x, y, width, height, img, publisher):
        detection = Detection2D()
        # Configure your Detection2D message
        # For example, setting the bounding box size and position
        detection.bbox.size_x = width
        detection.bbox.size_y = height
        detection.bbox.center.x = x + width // 2
        detection.bbox.center.y = y + height // 2
        # Convert OpenCV image to ROS image message
        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Attach image itself to the message as image_raw
        detection.source_img = ros_img
        # Publish the message
        publisher.publish(detection)

    def detect(self, image, frame_number):
        # Template matching
        x_d, y_d, width, height = timing_base_dynamic_template_match(image, self.template, self.prev, self.prev_prev)
        self.prev_prev = self.prev
        self.prev = (x_d, y_d, width, height)
        self.publish_detection(x_d, y_d, width, height, image, self.detected_pub)

        # Visualize groundtruth
        if frame_number < len(groundtruth):
            gt_x, gt_y, gt_w, gt_h = groundtruth[frame_number]
            self.publish_detection(gt_x, gt_y, gt_w, gt_h, image, self.groundtruth_pub)
        return image

def main():
    rospy.init_node('detector', anonymous=True)
    det = Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        det.stop_thread = True
        det.process_thread.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
