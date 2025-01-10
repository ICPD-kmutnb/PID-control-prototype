import cv2
from decouple import config
from Lib.prototype_class_tester import Dobot_PID_Control

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตรวจสอบว่ากล้องเปิดได้หรือไม่
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

# สร้าง instance ของ Dobot_PID_Control
image_process = Dobot_PID_Control()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # ประมวลผลภาพ
    result = image_process.get_result(ret, frame=frame)

    if result is not None:
        processed_frame, controls, errors, center = result
        print(f"Processed Frame: {processed_frame.shape}")  
        print(f"Controls: {controls}")
        print(f"Errors: {errors}")
        print(f"Center Point: {center}")

        """BlackBox"""
        #
        #
        #
        #
        #
        #
        """BlackBox"""

    
    cv2.imshow("Frame", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()