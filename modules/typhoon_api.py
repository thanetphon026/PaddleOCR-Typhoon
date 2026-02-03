"""
Typhoon API Integration - Intelligent Parcel Data Extraction
Extracts structured data from OCR text using Typhoon LLM
"""

import os
import json
import requests
from typing import Dict, Any

class TyphoonAPI:
    def __init__(self):
        """Initialize Typhoon API client"""
        self.api_key = os.getenv('TYPHOON_API_KEY', '')
        base_url = os.getenv('TYPHOON_API_URL', 'https://api.opentyphoon.ai/v1/chat/completions')
        
        # Ensure URL has /chat/completions endpoint
        if not base_url.endswith('/chat/completions'):
            self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        else:
            self.api_url = base_url
            
        self.model = os.getenv('TYPHOON_MODEL', 'typhoon-v2.5-30b-a3b-instruct')
        
        if not self.api_key:
            print("⚠ Warning: TYPHOON_API_KEY not set in .env file")

    
    def _create_extraction_prompt(self, ocr_text: str) -> str:
        """
        Create optimized prompt for Thai parcel data extraction
        
        Args:
            ocr_text: Raw OCR text from parcel image
            
        Returns:
            str: Formatted prompt for Typhoon API
        """
        prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์ข้อมูลพัสดุไทย จากข้อความที่สกัดได้จาก OCR กรุณาวิเคราะห์และสกัดข้อมูลต่อไปนี้:

1. **ชื่อผู้รับ** (recipient_name): ชื่อ-นามสกุล ของผู้รับพัสดุ
2. **เลขห้อง** (room_number): หมายเลขห้อง หรือที่อยู่ (ถ้ามี)
3. **บริษัทขนส่ง** (shipping_company): ชื่อบริษัทขนส่งพัสดุ (เช่น Kerry, Flash, Thailand Post, J&T, DHL, Ninja Van, Best Express, SCG Express, ไปรษณีย์ไทย)
4. **รหัสพัสดุ** (tracking_number): หมายเลขติดตามพัสดุ (Tracking Number)

**ข้อความจาก OCR:**
```
{ocr_text}
```

**คำแนะนำในการสกัดข้อมูล:**
- ถ้าไม่พบข้อมูลใด ให้ใส่ค่าว่าง "" หรือ "ไม่พบข้อมูล"
- รหัสพัสดุมักเป็นตัวเลขหรือตัวอักษรผสมกัน มักจะยาวประมาณ 10-20 ตัวอักษร
- บริษัทขนส่งมักจะมีโลโก้หรือชื่อบริษัทบนป้ายพัสดุ
- ชื่อผู้รับมักจะอยู่ในส่วน "ผู้รับ" หรือ "Receiver" หรือ "To"
- เลขห้องอาจอยู่ในที่อยู่ หรือเป็นตัวเลขแยกต่างหาก

**ตอบกลับเฉพาะ JSON เท่านั้น ในรูปแบบนี้:**
{{
    "recipient_name": "ชื่อผู้รับ",
    "room_number": "เลขห้อง",
    "shipping_company": "บริษัทขนส่ง",
    "tracking_number": "รหัสพัสดุ"
}}

**ห้าม** เพิ่มข้อความอื่นนอกเหนือจาก JSON"""
        
        return prompt
    
    def extract_parcel_data(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extract structured parcel data from OCR text using Typhoon API
        
        Args:
            ocr_text: Raw OCR text
            
        Returns:
            dict: Extracted parcel data with keys:
                - recipient_name
                - room_number
                - shipping_company
                - tracking_number
        """
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("Typhoon API key not configured")
            
            # Create prompt
            prompt = self._create_extraction_prompt(ocr_text)
            
            # Prepare API request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'คุณเป็นผู้เชี่ยวชาญในการสกัดข้อมูลจากข้อความ OCR ของพัสดุไทย ตอบเป็น JSON เท่านั้น'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.6,
                'max_completion_tokens': 512,
                'top_p': 0.6,
                'frequency_penalty': 0,
                'stream': False  # Set to True if you want streaming
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check response
            if response.status_code != 200:
                error_msg = f"Typhoon API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
                raise Exception(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract content from response
            if 'choices' not in result or len(result['choices']) == 0:
                raise Exception("Invalid response from Typhoon API")
            
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from response
            try:
                # Clean content - remove markdown code blocks if present
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Parse JSON
                extracted_data = json.loads(content)
                
                # Validate required fields
                required_fields = ['recipient_name', 'room_number', 'shipping_company', 'tracking_number']
                for field in required_fields:
                    if field not in extracted_data:
                        extracted_data[field] = "ไม่พบข้อมูล"
                
                return extracted_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw content: {content}")
                
                # Return default structure if parsing fails
                return {
                    "recipient_name": "ไม่สามารถสกัดข้อมูลได้",
                    "room_number": "ไม่สามารถสกัดข้อมูลได้",
                    "shipping_company": "ไม่สามารถสกัดข้อมูลได้",
                    "tracking_number": "ไม่สามารถสกัดข้อมูลได้",
                    "error": "Failed to parse JSON response"
                }
            
        except requests.exceptions.Timeout:
            raise Exception("Typhoon API request timeout (>30s)")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Typhoon API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Typhoon API error: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if Typhoon API is properly configured"""
        return bool(self.api_key and self.api_url)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test Typhoon API connection
        
        Returns:
            dict: Test result with status and message
        """
        try:
            test_text = "ผู้รับ: ทดสอบ ระบบ\nเลขห้อง: 101\nFlash Express\nTH123456789"
            result = self.extract_parcel_data(test_text)
            
            return {
                'status': 'success',
                'message': 'Typhoon API connection successful',
                'test_result': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
