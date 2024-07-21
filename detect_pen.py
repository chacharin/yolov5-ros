#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
import torch
import numpy as np
import cv2
from time import time
import os

class OBJ_Detection:
    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        
        model_path = os.path.expanduser('~/catkin_ws/src/my_pkg/scripts/best.pt')
        self.model = self.load_model(model_path)
        
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # ROS Initialization
        rospy.init_node('obj_detection')
        self.x_pub = rospy.Publisher('obj_val_x', Float32, queue_size=10)
        self.y_pub = rospy.Publisher('obj_val_y', Float32, queue_size=10)
        
    def load_model(self, model_path):     
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.15: # Threshold for confidence
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

                # Publish x and y coordinates of the first detected object
                self.x_pub.publish(x1)
                self.y_pub.publish(y1)
                break  # Only publish the first detected object

        return frame

    def class_to_label(self, x):
        return self.classes[int(x)]

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
      
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            assert ret
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1360, 670))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = OBJ_Detection(capture_index=0, model_name='best.pt')
    detector.run()


