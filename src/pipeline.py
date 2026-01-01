
import cv2
from .measurement import length_cm
from .biomass import weight

class Pipeline:
    def __init__(self, cam, det, k, a, b):
        self.cam,self.det,self.k,self.a,self.b = cam,det,k,a,b

    def run(self):
        while True:
            ok,frame = self.cam.read()
            if not ok: break
            boxes = self.det.detect(frame)
            total=0; n=0
            for b in boxes:
                L = length_cm(frame,b,self.k)
                W = weight(L,self.a,self.b)
                if W: total+=W; n+=1
                cv2.rectangle(frame,b[:2],b[2:],(0,255,0),2)
            cv2.putText(frame,f"Count:{n} Biomass:{total:.1f}g",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imshow("Shrimp Biomass",frame)
            if cv2.waitKey(1)==27: break
        self.cam.release(); cv2.destroyAllWindows()
