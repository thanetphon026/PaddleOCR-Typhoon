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
        # ดึงค่าจาก .env
        self.api_key = os.getenv('TYPHOON_API_KEY', '')
        # แก้ไขจุดนี้: ถ้ามี URL ใน .env ให้ใช้ตามนั้น ถ้าไม่มีให้ใช้ Default ของ Typhoon
        raw_url = os.getenv('TYPHOON_API_URL', 'https://api.opentyphoon.ai/v1')
        
        # จัดการ URL ให้ถูกต้อง (ป้องกันการใส่ /chat/completions ซ้ำซ้อน)
        base_url = raw_url.rstrip('/')
        if not base_url.endswith('/chat/completions'):
            self.api_url = f"{base_url}/chat/completions"
        else:
            self.api_url = base_url
            
        self.model = os.getenv('TYPHOON_MODEL', 'typhoon-v2.5-30b-a3b-instruct')
        
        if not self.api_key:
            print("⚠ Warning: TYPHOON_API_KEY not set in .env file")

    def _create_extraction_prompt(self, ocr_text: str) -> str:
        """Create optimized prompt for Thai parcel data extraction"""
        prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์ข้อมูลพัสดุไทย จากข้อความที่สกัดได้จาก OCR กรุณาวิเคราะห์และสกัดข้อมูลต่อไปนี้ในรูปแบบ JSON:

1. **ชื่อผู้รับ** (recipient_name)
2. **เลขห้อง** (room_number)
3. **บริษัทขนส่ง** (shipping_company)
4. **รหัสพัสดุ** (tracking_number)

**ข้อความจาก OCR:**
{ocr_text}

**ตอบกลับเฉพาะ JSON เท่านั้น ห้ามมีคำอธิบายอื่น**"""
        return prompt
    
    def extract_parcel_data(self, ocr_text: str) -> Dict[str, Any]:
        try:
            if not self.api_key:
                raise ValueError("Typhoon API key not configured")
            
            prompt = self._create_extraction_prompt(ocr_text)
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key.strip()}'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': 'คุณเป็นผู้เชี่ยวชาญด้านข้อมูลพัสดุ ตอบกลับเป็น JSON เท่านั้น'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.1, # ปรับให้คำตอบนิ่งที่สุด
                'max_tokens': 512
            }
            
            # ยิง API
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # ถ้าเจอ 400 หรืออื่นๆ จะโชว์รายละเอียดที่แท้จริงจาก Typhoon
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_detail = response.json().get('error', {}).get('message', response.text)
                except: pass
                raise Exception(f"HTTP {response.status_code}: {error_detail}")
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # ทำความสะอาด JSON (เผื่อ LLM ใส่ ```json มา)
            content = content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            return json.loads(content)
                
        except Exception as e:
            print(f"DEBUG - API URL: {self.api_url}") # ช่วยดูว่า URL ผิดไหม
            raise Exception(f"Typhoon API error: {str(e)}")

    def is_configured(self) -> bool:
        return bool(self.api_key)