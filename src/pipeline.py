
import cv2
from .geometry import PlaneScale
from .measurement import measure_length
from .biomass import weight_g

class Pipeline:
    def __init__(self, cam, det, k, a, b):
        self.cam,self.det,self.a,self.b = cam,det,a,b
        self.scale = PlaneScale(k)

    def run(self):
        while True:
            ok,frame = self.cam.read()
            if not ok: break
            boxes = self.det.detect(frame)
            total=0; n=0
            for det in boxes:
                b = (det.x1, det.y1, det.x2, det.y2)
                L = measure_length(frame,det,self.scale,{}).length_cm
                W = weight_g(L,self.a,self.b)
                if W: total+=W; n+=1
                cv2.rectangle(frame,b[:2],b[2:],(0,255,0),2)
            cv2.putText(frame,f"Count:{n} Biomass:{total:.1f}g",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imshow("Shrimp Biomass",frame)
            if cv2.waitKey(1)==27: break
        self.cam.release(); cv2.destroyAllWindows()
