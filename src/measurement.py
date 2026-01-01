
import cv2

def length_cm(frame, box, k):
    x1,y1,x2,y2 = box
    roi = frame[y1:y2,x1:x2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(g,0,255,cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    (_, _),(w,h),_ = cv2.minAreaRect(max(cnts,key=cv2.contourArea))
    return max(w,h)*k
