import cv2
import numpy as np
from decouple import config
import time
from PID_Control_class import PIDController


# เปิดกล้อง
cap = cv2.VideoCapture(0)

# Center point
Center_point = [0, 0]

# PID สำหรับแกน X และ Y
pid_x = PIDController(kp=float(config("Kp")), 
                      ki=float(config("Ki")), 
                      kd=float(config("Kd")))

pid_y = PIDController(kp=float(config("Kp")), 
                      ki=float(config("Ki")), 
                      kd=float(config("Kd")))

# เวลาเริ่มต้น
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ขนาดของภาพ
    height, width, _ = frame.shape
    Center_point[0] = width // 2
    Center_point[1] = height // 2

    # สีเส้น
    Line_color = (int(config("B")), int(config("G")), int(config("R")))

    # วาดเส้นกลาง
    cv2.line(frame, (Center_point[0], 0), (Center_point[0], height), Line_color, thickness=2)
    cv2.line(frame, (0, Center_point[1]), (width, Center_point[1]), Line_color, thickness=2)

    # แปลงภาพเป็น HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # สีเป้าหมาย
    lower_color = np.array([int(config("LOWER_H")), 
                            int(config("LOWER_S")), 
                            int(config("LOWER_V"))])
    
    upper_color = np.array([int(config("UPPER_H")), 
                            int(config("UPPER_S")), 
                            int(config("UPPER_V"))])

    # สร้าง mask และลด noise
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # หา contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

    # คำนวณค่า PID
    if len(centroids) > 0:
        avg_cx = int(np.mean([c[0] for c in centroids]))
        avg_cy = int(np.mean([c[1] for c in centroids]))

        # คำนวณ error
        error_x = Center_point[0] - avg_cx
        error_y = Center_point[1] - avg_cy

        # เวลาปัจจุบัน
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # คำนวณ PID output
        control_x = pid_x.compute(error_x, dt)
        control_y = pid_y.compute(error_y, dt)

        # วาด centroid และ error
        cv2.circle(frame, (avg_cx, avg_cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Error X: {error_x}, Control X: {control_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Error Y: {error_y}, Control Y: {control_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print("CenterPoint_frame    :",Center_point[0],Center_point[1])
        print("Centroid             :",avg_cx,avg_cy)
        print("Control              :",control_x,control_y)
    # แสดงผล
    cv2.imshow("Frame", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()