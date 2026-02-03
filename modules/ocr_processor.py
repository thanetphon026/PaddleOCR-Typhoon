"""
PaddleOCR Processor - Optimized for Thai Parcel Scanning
Supports both CPU and GPU with automatic detection
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import os

class OCRProcessor:
    def __init__(self, force_cpu=True):
        """Initialize PaddleOCR with optimized settings for Thai+English text"""
        self.ocr = None
        self.use_gpu = False if force_cpu else self._check_gpu_availability()
        self._initialize_ocr()
    
    def _check_gpu_availability(self):
        """Check if GPU is available for PaddleOCR"""
        try:
            import paddle
            has_cuda = paddle.is_compiled_with_cuda()
            
            if has_cuda:
                # Additional check: try to see if CUDA libraries are accessible
                try:
                    # This will fail if CUDA/cuDNN is not properly installed
                    import paddle.fluid as fluid
                    return True
                except:
                    print("⚠ CUDA detected but libraries not accessible, using CPU mode")
                    return False
            return False
        except Exception as e:
            print(f"⚠ GPU check failed: {str(e)}, using CPU mode")
            return False
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR engine with optimal settings"""
        try:
            print(f"Initializing PaddleOCR (Mode: {'GPU' if self.use_gpu else 'CPU'})...")
            
            # Try to initialize with Thai settings using PP-OCRv3 (which supports 'th')
            try:
                self.ocr = PaddleOCR(
                    # Language settings
                    lang='th',  # Thai language
                    ocr_version='PP-OCRv3',  # Specific version that supports Thai fully
                    
                    # GPU/CPU settings
                    use_gpu=self.use_gpu,
                    gpu_mem=500 if self.use_gpu else 0,
                    
                    # Performance optimization
                    use_angle_cls=True,
                    use_mp=True,
                    total_process_num=2,
                    
                    # CPU optimization
                    enable_mkldnn=True if not self.use_gpu else False,
                    
                    # Model settings
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6,
                    
                    # Accuracy settings
                    drop_score=0.3,
                    
                    # Display settings
                    show_log=False,
                    use_space_char=True
                )
                
                print(f"✓ PaddleOCR initialized successfully (Mode: {'GPU' if self.use_gpu else 'CPU'}, Lang: Thai)")
                
            except Exception as e:
                # If Thai initialization fails, try fallback to Chinese (ch) which contains some support
                # or handle specific errors
                print(f"⚠ Thai OCR initialization failed: {str(e)}")
                print("→ Falling back to general multi-language model (Chinese/English)...")
                
                self.ocr = PaddleOCR(
                    lang='ch',  # Fallback to Chinese/English
                    use_gpu=self.use_gpu,
                    use_angle_cls=True,
                    use_mp=True,
                    total_process_num=2,
                    enable_mkldnn=True if not self.use_gpu else False,
                    show_log=False
                )
                print(f"✓ PaddleOCR initialized with fallback (Mode: {'GPU' if self.use_gpu else 'CPU'}, Lang: CH/EN)")
                
            except RuntimeError as e:
                # If GPU initialization fails (CUDA/cuDNN issue), fallback to CPU
                if 'cudnn' in str(e).lower() or 'cuda' in str(e).lower():
                    print(f"⚠ GPU initialization failed: {str(e)}")
                    print("→ Falling back to CPU mode...")
                    
                    self.use_gpu = False
                    
                    # Retry with CPU
                    self.ocr = PaddleOCR(
                        lang='th',  # Thai language for CPU mode
                        use_gpu=False,
                        use_angle_cls=True,
                        use_mp=True,
                        total_process_num=2,
                        enable_mkldnn=True,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                        rec_batch_num=6,
                        drop_score=0.3,
                        show_log=False,
                        use_space_char=True
                    )
                    
                    print(f"✓ PaddleOCR initialized successfully (CPU mode)")
                else:
                    raise
            
        except Exception as e:
            print(f"✗ Error initializing PaddleOCR: {str(e)}")
            raise

    
    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR accuracy
        - Resize if too large
        - Enhance contrast
        - Denoise
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Resize if image is too large (max 2000px on longest side)
            height, width = img.shape[:2]
            max_dim = 2000
            
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better contrast
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            processed = cv2.fastNlMeansDenoising(processed, h=10)
            
            # Save preprocessed image temporarily
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, processed)
            
            return temp_path
            
        except Exception as e:
            print(f"Warning: Preprocessing failed, using original image: {str(e)}")
            return image_path
    
    def extract_text(self, image_path):
        """
        Extract text from parcel image using PaddleOCR
        
        Args:
            image_path: Path to parcel image
            
        Returns:
            str: Extracted text content
        """
        try:
            # Preprocess image
            processed_path = self.preprocess_image(image_path)
            
            # Run OCR
            result = self.ocr.ocr(processed_path, cls=True)
            
            # Clean up preprocessed image
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)
            
            # Extract text from results
            if not result or not result[0]:
                return ""
            
            # Combine all detected text with confidence filtering
            text_lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # Extract text
                    confidence = line[1][1]  # Extract confidence
                    
                    # Only include text with reasonable confidence
                    if confidence > 0.3:
                        text_lines.append(text)
            
            # Join all text lines
            full_text = '\n'.join(text_lines)
            
            return full_text.strip()
            
        except Exception as e:
            print(f"✗ OCR extraction error: {str(e)}")
            raise
    
    def is_ready(self):
        """Check if OCR processor is ready"""
        return self.ocr is not None
    
    def get_device_info(self):
        """Get information about current processing device"""
        return {
            'gpu_available': self.use_gpu,
            'device': 'GPU' if self.use_gpu else 'CPU'
        }
