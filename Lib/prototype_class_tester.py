import cv2
import numpy as np
from decouple import config
import time
from Lib.PID_Control_class import PIDController

class Dobot_PID_Control:
    def __init__(self):
        self.Center_point = [0, 0]
        self.pid_x = PIDController(kp=float(config("Kp")), 
                                    ki=float(config("Ki")), 
                                    kd=float(config("Kd")))
        self.pid_y = PIDController(kp=float(config("Kp")), 
                                    ki=float(config("Ki")), 
                                    kd=float(config("Kd")))
        self.last_time = time.time()

    def get_result(self, ret, frame):
        if not ret:
            print("No frame received.")
            return None

        print("Processing frame...")

        height, width, _ = frame.shape
        self.Center_point[0] = (width // 2)+int(config("ShiftVertical"))                   #ขยับcenterแนวตั้ง
        self.Center_point[1] = (height // 2)+int(config("ShiftHorizontal"))                #ขยับcenterแนวนอน

        Line_color = (int(config("B")), int(config("G")), int(config("R")))

        cv2.line(frame, (self.Center_point[0], 0), (self.Center_point[0], height), Line_color, thickness=2)     #แนวตั้ง
        cv2.line(frame, (0, self.Center_point[1]), (width, self.Center_point[1]), Line_color, thickness=2)      #แนวนอน

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_color = np.array([int(config("LOWER_H")), 
                                int(config("LOWER_S")), 
                                int(config("LOWER_V"))])
        
        upper_color = np.array([int(config("UPPER_H")), 
                                int(config("UPPER_S")), 
                                int(config("UPPER_V"))])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if len(centroids) > 0:
            avg_cx = int(np.mean([c[0] for c in centroids]))
            avg_cy = int(np.mean([c[1] for c in centroids]))

            error_x = self.Center_point[0] - avg_cx
            error_y = self.Center_point[1] - avg_cy

            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            control_x = self.pid_x.compute(error_x, dt)
            control_y = self.pid_y.compute(error_y, dt)

            cv2.circle(frame, (avg_cx, avg_cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"Error X: {error_x}, Control X: {control_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Error Y: {error_y}, Control Y: {control_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            print(f"CenterPoint_frame: {self.Center_point[0]}, {self.Center_point[1]}")
            print(f"Centroid: {avg_cx}, {avg_cy}")
            print(f"Control: {control_x}, {control_y}")

            return frame, [control_x, control_y], [error_x, error_y], [self.Center_point[0], self.Center_point[1]]
        else:
            print("No centroids found.")
            return None