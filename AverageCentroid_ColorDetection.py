import cv2
import numpy as np

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break
    
    # แปลงภาพเป็น HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงสีเขียวใน HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # สร้าง mask สำหรับสีเขียว
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # ลด noise ด้วยการใช้ morphology (เปิดและปิด)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # ใช้ bitwise_and เพื่อแสดงเฉพาะส่วนที่เป็นสีเขียว
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # หา contours ของ mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # เก็บจุด centroid ทั้งหมด
    centroids = []
    
    for contour in contours:
        # กรอง contours เล็ก ๆ ออก
        if cv2.contourArea(contour) > 500:
            # หา moments เพื่อคำนวณ centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    
    # คำนวณ centroid โดยรวมจาก centroid ทั้งหมด
    if len(centroids) > 0:
        avg_cx = int(np.mean([c[0] for c in centroids]))
        avg_cy = int(np.mean([c[1] for c in centroids]))
        
        # วาด centroid รวม
        cv2.circle(frame, (avg_cx, avg_cy), 5, (255, 0, 0), -1)
    
    # แสดงผล
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    
    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()