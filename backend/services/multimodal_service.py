"""Production Multimodal Processing Service"""
import io
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
import asyncio

from PIL import Image
import librosa
import soundfile as sf
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration
)
import torch
import PyPDF2
from docx import Document

from backend.config import settings
from backend.utils.logger import log

class MultimodalService:
    """Service for processing multimodal inputs"""
    
    def __init__(self):
        self.whisper_model = None
        self.whisper_processor = None
        self.vision_model = None
        self.vision_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize multimodal models"""
        log.info("Initializing multimodal service...")
        
        # Initialize audio model (Whisper)
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                "openai/whisper-base",
                cache_dir=str(settings.MODELS_DIR)
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-base",
                cache_dir=str(settings.MODELS_DIR)
            ).to(self.device)
            log.info("✓ Whisper model loaded")
        except Exception as e:
            log.error(f"Failed to load Whisper: {str(e)}")
        
        # Initialize vision model (BLIP)
        try:
            self.vision_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=str(settings.MODELS_DIR)
            )
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=str(settings.MODELS_DIR)
            ).to(self.device)
            log.info("✓ Vision model loaded")
        except Exception as e:
            log.error(f"Failed to load vision model: {str(e)}")
        
        log.info("✓ Multimodal service initialized")
    
    async def process_image(
        self,
        image_data: Union[bytes, str, Path],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process an image and generate description or answer query"""
        
        log.info("Processing image...")
        
        try:
            # Load image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str) and image_data.startswith("data:image"):
                # Base64 encoded image
                base64_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(image_data)
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process with vision model
            if query:
                # Visual question answering
                inputs = self.vision_processor(image, query, return_tensors="pt").to(self.device)
            else:
                # Image captioning
                inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                output = self.vision_model.generate(**inputs, max_length=100)
            
            # Decode output
            result = self.vision_processor.decode(output[0], skip_special_tokens=True)
            
            log.info("✓ Image processed successfully")
            
            return {
                "success": True,
                "result": result,
                "type": "image_caption" if not query else "visual_qa"
            }
            
        except Exception as e:
            log.error(f"Image processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_audio(
        self,
        audio_data: Union[bytes, str, Path],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process audio and transcribe"""
        
        log.info("Processing audio...")
        
        try:
            # Load audio
            if isinstance(audio_data, bytes):
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_data))
            else:
                audio_array, sampling_rate = librosa.load(audio_data, sr=16000)
            
            # Process with Whisper
            inputs = self.whisper_processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            # Move to device
            input_features = inputs.input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    language=language
                )
            
            # Decode transcription
            transcription = self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            log.info("✓ Audio processed successfully")
            
            return {
                "success": True,
                "transcription": transcription,
                "duration": len(audio_array) / sampling_rate
            }
            
        except Exception as e:
            log.error(f"Audio processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        operation: str = "extract"
    ) -> Dict[str, Any]:
        """Process document (PDF, DOCX, etc.)"""
        
        log.info(f"Processing document: {file_path}")
        
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == ".pdf":
                text = self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() == ".docx":
                text = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_path.suffix}"
                }
            
            # Perform operation
            if operation == "extract":
                result = text
            elif operation == "summarize":
                # Would use a summarization model here
                result = text[:500] + "..." if len(text) > 500 else text
            else:
                result = text
            
            log.info("✓ Document processed successfully")
            
            return {
                "success": True,
                "text": result,
                "length": len(text),
                "operation": operation
            }
            
        except Exception as e:
            log.error(f"Document processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])