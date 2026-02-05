from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

class OCRProcessor:
    def __init__(self, force_cpu=False):
        self.ocr = None
        # บังคับเช็ค GPU
        self.use_gpu = False if force_cpu else self._check_gpu_availability()
        self._initialize_ocr()
    
    def _check_gpu_availability(self):
        try:
            import paddle
            # เช็คทั้งการ Compile และการมองเห็นตัวการ์ดจอ
            return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except:
            return False
    
    def _initialize_ocr(self):
        try:
            print(f"🚀 Initializing PaddleOCR (Device: {'GPU' if self.use_gpu else 'CPU'})...")
            
            # แก้ปัญหา AssertionError: 
            # สำหรับ PP-OCRv4 บน Windows ให้ใช้ lang='latin' 
            # เพราะโมเดล Latin ของ V4 ครอบคลุมภาษาไทย (Thai) ไว้ข้างในแล้ว
            self.ocr = PaddleOCR(
                use_gpu=self.use_gpu,
                lang='latin',              # ** ห้ามเปลี่ยนเป็น 'th' เพราะ V4 จะ Error **
                ocr_version='PP-OCRv4',    # ใช้ตัวล่าสุดที่ฉลาดที่สุด
                use_angle_cls=True,        # ตรวจจับข้อความเอียง
                show_log=False,
                rec_batch_num=6,
                enable_mkldnn=True if not self.use_gpu else False
            )
            
            print(f"✅ PaddleOCR Ready! [Mode: {'GPU' if self.use_gpu else 'CPU'}]")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """ ปรับภาพให้คมชัดขึ้นก่อนสแกน """
        try:
            img = cv2.imread(image_path)
            if img is None: return image_path
            
            # ขยายภาพ 2 เท่าเพื่อให้ตัวอักษรไทยชัดขึ้น
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # ปรับ Contrast
            alpha = 1.5 
            beta = 0   
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            temp_path = image_path.replace('.', '_proc.')
            cv2.imwrite(temp_path, img)
            return temp_path
        except:
            return image_path
    
    def extract_text(self, image_path):
        processed_path = None
        try:
            processed_path = self.preprocess_image(image_path)
            # รัน OCR โดยเน้นภาษาไทย
            result = self.ocr.ocr(processed_path, cls=True)
            
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)
            
            if not result or not result[0]: return ""
            
            return '\n'.join([line[1][0] for line in result[0] if line[1][1] > 0.4]).strip()
            
        except Exception as e:
            if processed_path and os.path.exists(processed_path): os.remove(processed_path)
            raise e
    def get_device_info(self):
        """ส่งค่าคืนให้ app.py ตามที่มันต้องการ"""
        return {
            'gpu_available': self.use_gpu,
            'device': 'GPU' if self.use_gpu else 'CPU'
        }
    def is_ready(self):
        """ส่งค่าคืนให้ app.py เพื่อเช็คสถานะ"""
        return self.ocr is not None

    def get_device_info(self):
        """ส่งค่าคืนให้ app.py เพื่อโชว์สถานะ GPU/CPU"""
        return {
            'gpu_available': self.use_gpu,
            'device': 'GPU' if self.use_gpu else 'CPU'
        }