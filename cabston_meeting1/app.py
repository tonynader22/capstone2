import os
import re
import time
import psutil
import shutil
import warnings
import tempfile
import subprocess
import hashlib
import json
import requests
from pathlib import Path
from functools import lru_cache, wraps
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import List, Dict, Union
from langdetect import detect
from faster_whisper import WhisperModel
import onnxruntime as ort
from pydub import AudioSegment
import tiktoken

# ============== تحميل متغيرات البيئة من ملف .env (إن وجد) ==============
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # إذا لم تتوفر dotenv تجاهلها فقط

# ============== إعداد مفاتيح API والمتغيرات الحساسة ==============
# يجب تعيين هذه المتغيرات في متغيرات البيئة أو ملف .env
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # مفتاح Groq API
username = os.environ.get('USERNAME')  # اسم المستخدم ثابت
password = os.environ.get('PASSWORD')  # كلمة المرور ثابتة
SECRET_KEY = os.environ.get('SECRET_KEY')  # مفتاح الجلسة (يجب تعيينه في .env)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"
GROQ_RATE_LIMIT_DELAY = 3  # تأخير أولي 3 ثوانٍ
GROQ_MAX_RETRIES = 5
GROQ_MAX_BACKOFF = 30  # الحد الأقصى للتأخير 30 ثانية

# ============== التهيئة والإعدادات ==============

app = Flask(__name__)
CORS(app)
app.secret_key = SECRET_KEY  # مفتاح الجلسة

# إعدادات النظام
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
SUPPORTED_LANGUAGES = {"الإنجليزية": "en", "العربية": "ar"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_RETRIES = 3
RETRY_DELAY = 1

# إعدادات معالجة الملفات الكبيرة
MAX_CHUNK_SIZE_MB = 50  # الحد الأقصى لحجم الجزء بالميجابايت
MIN_FREE_DISK_SPACE_MB = 1000  # الحد الأدنى للمساحة الحرة المطلوبة بالميجابايت
MIN_FREE_MEMORY_MB = 500  # الحد الأدنى للذاكرة الحرة المطلوبة بالميجابايت
TEMP_DIR = Path(tempfile.gettempdir()) / "audio_processing"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# إعدادات التخزين المؤقت
CACHE_DIR = Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

# تجاهل تحذيرات غير ضرورية
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=UserWarning)

# ============== نظام تسجيل الأخطاء ==============

class ErrorLogger:
    @staticmethod
    def log_error(error_type, message, details=None):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": error_type,
            "message": message,
            "details": details
        }
        print(f"❌ [ERROR] {json.dumps(log_entry, ensure_ascii=False)}")
        
        # حفظ السجل في ملف
        log_file = Path("error_logs.json")
        try:
            if log_file.exists():
                with open(log_file, "r+", encoding="utf-8") as f:
                    logs = json.load(f)
                    logs.append(log_entry)
                    f.seek(0)
                    json.dump(logs, f, ensure_ascii=False, indent=2)
            else:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump([log_entry], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ فشل حفظ السجل: {str(e)}")

class ResourceManager:
    """فئة لإدارة موارد النظام"""
    
    @staticmethod
    def check_system_resources():
        """التحقق من موارد النظام المتاحة"""
        try:
            # التحقق من المساحة الحرة
            free_disk = shutil.disk_usage(TEMP_DIR).free / (1024 * 1024)  # بالميجابايت
            if free_disk < MIN_FREE_DISK_SPACE_MB:
                raise ResourceWarning(f"مساحة القرص غير كافية. المتاح: {free_disk:.0f}MB")
            
            # التحقق من الذاكرة المتاحة
            free_memory = psutil.virtual_memory().available / (1024 * 1024)  # بالميجابايت
            if free_memory < MIN_FREE_MEMORY_MB:
                raise ResourceWarning(f"الذاكرة غير كافية. المتاح: {free_memory:.0f}MB")
            
            return True
        except Exception as e:
            ErrorLogger.log_error("ResourceCheck", str(e))
            return False
    
    @staticmethod
    def cleanup_temp_files(older_than_hours=24):
        """تنظيف الملفات المؤقتة القديمة"""
        try:
            current_time = time.time()
            for temp_file in TEMP_DIR.glob("*"):
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > older_than_hours * 3600:
                        temp_file.unlink(missing_ok=True)
        except Exception as e:
            ErrorLogger.log_error("Cleanup", "فشل تنظيف الملفات المؤقتة", str(e))

class RetryManager:
    """فئة لإدارة إعادة المحاولات"""
    
    @staticmethod
    def with_retry(func, *args, max_retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
        """تنفيذ الدالة مع إعادة المحاولة في حالة الفشل"""
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(delay * attempt)  # تأخير تصاعدي
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                ErrorLogger.log_error(
                    "RetryAttempt",
                    f"فشلت المحاولة {attempt + 1}/{max_retries}",
                    str(e)
                )
        raise last_error

# ============== فئات المصنع مع التحسينات ==============

class ModelLoader:
    """فئة لتحميل النماذج مع دعم GPU/CPU التلقائي"""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._device = cls._instance._select_device()
            cls._instance._onnx_sessions = {}
            
            print(f"⚡ الجهاز المستخدم: {cls._instance._device}")
            
            if cls._instance._device.type == 'cpu':
                ort.set_default_logger_severity(3)
        return cls._instance
    
    def _select_device(self):
        """اختيار الجهاز الأمثل"""
        try:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                print("CUDA متاح، سيتم استخدامه")
                return type("Device", (), {"type": "cuda"})()
        except Exception:
            pass
        return type("Device", (), {"type": "cpu"})()
    
    def preload_models(self):
        """تحميل مسبق لجميع النماذج"""
        if not self._models_loaded:
            print("⚡ جارٍ تحميل النماذج مسبقاً...")
            start_time = time.time()
            
            try:
                self._load_onnx_models()
            except Exception as e:
                ErrorLogger.log_error("ModelLoad", "تعذر تحميل نماذج ONNX", str(e))
            
            self._models_loaded = True
            print(f"✅ تم تحميل جميع النماذج في {time.time() - start_time:.2f} ثانية")
    
    def _load_onnx_models(self):
        """تحميل نماذج ONNX المحسنة"""
        pass
    
    def get_whisper_model(self):
        if 'whisper' not in self._models:
            print("⚡ جارٍ تحميل نموذج Faster Whisper...")
            start_time = time.time()
            try:
                compute_type = "float16" if self._device.type == 'cuda' else "float32"
                device = "cuda" if self._device.type == 'cuda' else "cpu"
                
                model = WhisperModel(
                    "small",
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=min(4, os.cpu_count() or 1) if device == "cpu" else 0,
                    num_workers=1
                )
                
                self._models['whisper'] = model
                print(f"✅ تم تحميل Faster Whisper في {time.time() - start_time:.2f} ثانية")
            except Exception as e:
                ErrorLogger.log_error("ModelLoad", "فشل تحميل Faster Whisper", str(e))
                return None
        return self._models['whisper']
    

model_loader = ModelLoader()

# ============== نظام التخزين المؤقت المحسن ==============

class AudioCache:
    """فئة لإدارة التخزين المؤقت مع تنظيم الملفات"""
    
    @staticmethod
    def get_file_hash(file_path):
        """حساب hash متقدم للملف"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def get_cache_path(file_hash):
        """تنظيم الملفات المؤقتة في مجلدات فرعية"""
        subdir = CACHE_DIR / file_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{file_hash}.json"
    
    @staticmethod
    def check_cache(file_path):
        """التحقق من التخزين المؤقت مع التحقق من صلاحية الملف"""
        file_hash = AudioCache.get_file_hash(file_path)
        cache_file = AudioCache.get_cache_path(file_hash)
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    
                    if (os.path.getsize(file_path) == cache_data["file_size"] and 
                        os.path.getmtime(file_path) == cache_data["last_modified"]):
                        return cache_data["transcription"]
            except Exception as e:
                ErrorLogger.log_error("CacheError", "فشل قراءة الملف المؤقت", str(e))
        return None
    
    @staticmethod
    def save_to_cache(file_path, transcription):
        """حفظ التفريغ الصوتي في التخزين المؤقت"""
        file_hash = AudioCache.get_file_hash(file_path)
        cache_file = AudioCache.get_cache_path(file_hash)
        
        cache_data = {
            "file_hash": file_hash,
            "file_size": os.path.getsize(file_path),
            "last_modified": os.path.getmtime(file_path),
            "transcription": transcription,
            "timestamp": time.time()
        }
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            ErrorLogger.log_error("CacheError", "فشل حفظ الملف المؤقت", str(e))
        
        return cache_data

# ============== دوال المساعدة المحسنة ==============

class TextProcessor:
    """فئة لمعالجة النصوص مع تحسينات للأداء"""
    
    @staticmethod
    @lru_cache(maxsize=500)
    def clean_text(text):
        """تنظيف النص مع تحسينات للغة العربية"""
        if not text:
            return ""
        
        text = re.sub(r'[.؟!،؛]+', '.', text)
        text = re.sub(r'(\S+)( \1)+', r'\1', text)
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return ""
        
        if TextProcessor.detect_language_safe(text) == "ar":
            sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        else:
            sentences = [s.capitalize() for s in sentences]
        
        return '. '.join(sentences) + '.' if sentences else ''

    @staticmethod
    @lru_cache(maxsize=100)
    def detect_language_safe(text):
        """كشف اللغة مع التعامل مع الأخطاء"""
        if not text:
            return "en"
        try:
            lang = detect(text[:500])
            return "ar" if lang == "ar" else "en"
        except Exception as e:
            ErrorLogger.log_error("LanguageDetection", "فشل كشف اللغة", str(e))
            return "en"

    @staticmethod
    def fast_text_split(text, max_chars=5000):
        """تقسيم سريع للنص مع الحفاظ على الكلمات"""
        if len(text) <= max_chars:
            return [text]
        split_pos = text.rfind(' ', 0, max_chars)
        return [text[:split_pos]] + TextProcessor.fast_text_split(text[split_pos+1:], max_chars)

class FileProcessor:
    """فئة لمعالجة الملفات الصوتية فقط"""
    @staticmethod
    def process_uploaded_file(file):
        if not file:
            print("===> process_uploaded_file: No file object received")
            return None, "لم يتم تقديم ملف"
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        filename = secure_filename(file.filename)
        file_ext = filename.lower().split('.')[-1]
        print(f"===> process_uploaded_file: filename={filename}, size={file_size}, ext={file_ext}")
        if file_size > MAX_FILE_SIZE:
            print(f"===> process_uploaded_file: File too large: {file_size}")
            return None, "حجم الملف يتجاوز الحد المسموح (5GB)"
        # السماح فقط بامتدادات الصوت
        if file_ext not in ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm']:
            return None, "نوع الملف غير مدعوم، فقط الملفات الصوتية مسموحة"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
                file.save(temp_file.name)
                print(f"===> process_uploaded_file: Saved to temp: {temp_file.name}")
                return temp_file.name, None
        except Exception as e:
            ErrorLogger.log_error("FileUpload", "فشل معالجة الملف المرفوع", str(e))
            print(f"===> process_uploaded_file: Exception: {str(e)}")
            return None, f"خطأ في معالجة الملف: {str(e)}"

    @staticmethod
    def process_audio_file(file_path):
        try:
            print(f"===> process_audio_file: file_path={file_path}")
            cached_text = AudioCache.check_cache(file_path)
            if cached_text:
                print(f"⚡ تم استعادة التفريغ من التخزين المؤقت")
                return cached_text
            # استخدم Faster Whisper المحلي
            text = AudioProcessor.recognize_speech(file_path)
            if not text:
                raise ValueError("لم يتم استخراج نص من الملف الصوتي")
            AudioCache.save_to_cache(file_path, text)
            return text
        except Exception as e:
            ErrorLogger.log_error("AudioProcessing", "فشل معالجة الملف الصوتي", str(e))
            print(f"===> process_audio_file: Exception: {str(e)}")
            return f"خطأ في معالجة الملف الصوتي: {str(e)}"
        finally:
            if file_path.endswith('_optimized.wav'):
                try:
                    os.remove(file_path)
                except:
                    pass
    
class AudioProcessor:
    """فئة متخصصة لمعالجة الملفات الصوتية"""
    
    @staticmethod
    def validate_audio_file(audio_path):
        """التحقق من صحة الملف الصوتي"""
        try:
            # التحقق من وجود الملف
            if not os.path.exists(audio_path):
                raise ValueError("الملف غير موجود")
            
            # التحقق من حجم الملف
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError("الملف فارغ")
            
            # التحقق من صحة الملف الصوتي
            audio = AudioSegment.from_file(audio_path)
            if len(audio) == 0:
                raise ValueError("الملف الصوتي فارغ")
            if audio.duration_seconds < 0.1:
                raise ValueError("الملف الصوتي قصير جداً")
            
            return True
        except Exception as e:
            ErrorLogger.log_error("AudioValidation", "فشل التحقق من صحة الملف الصوتي", str(e))
            return False
    
    @staticmethod
    def optimize_audio_for_whisper(input_path, output_path):
        """تحسين إعدادات الصوت لزيادة سرعة Whisper"""
        try:
            if not ResourceManager.check_system_resources():
                raise ResourceWarning("موارد النظام غير كافية لمعالجة الصوت")
            
            if not AudioProcessor.validate_audio_file(input_path):
                raise ValueError("الملف الصوتي غير صالح")
            
            command = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-y',
                '-hide_banner',
                '-loglevel', 'error',
                output_path
            ]
            
            result = RetryManager.with_retry(
                subprocess.run,
                command,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"فشل تحويل الصوت: {result.stderr}")
            
            # التحقق من الملف الناتج
            if not AudioProcessor.validate_audio_file(output_path):
                raise ValueError("فشل تحسين الملف الصوتي")
            
            return True
        except Exception as e:
            ErrorLogger.log_error("AudioOptimization", "فشل تحسين الصوت", str(e))
            return False
    
    @staticmethod
    def recognize_speech(audio_path):
        """التعرف على الكلام في ملف صوتي"""
        if not AudioProcessor.validate_audio_file(audio_path):
            return "الملف الصوتي غير صالح"
        
        model = model_loader.get_whisper_model()
        if not model:
            return "نموذج التعرف على الكلام غير متاح"
        
        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # تجميع النص من جميع المقاطع
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            if not transcription:
                raise ValueError("لم يتم التعرف على أي نص")
            
            AudioCache.save_to_cache(audio_path, transcription)
            return transcription
        except Exception as e:
            ErrorLogger.log_error("SpeechRecognition", "فشل التعرف على الكلام", str(e))
            return f"خطأ في معالجة الملف: {str(e)}"
    
    @staticmethod
    def process_large_audio(audio_path, chunk_size=MAX_CHUNK_SIZE_MB):
        """تجزئة ومعالجة الملفات الصوتية الكبيرة"""
        if not ResourceManager.check_system_resources():
            raise ResourceWarning("موارد النظام غير كافية لمعالجة الملف الكبير")
        
        if not AudioProcessor.validate_audio_file(audio_path):
            raise ValueError("الملف الصوتي غير صالح")
        
        model = model_loader.get_whisper_model()
        if not model:
            raise RuntimeError("نموذج التعرف على الكلام غير متاح")
        
        try:
            # تحويل الحجم من ميجابايت إلى ميللي ثانية
            chunk_duration = chunk_size * 1000  # تحويل إلى ميللي ثانية
            
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio)
            
            if duration == 0:
                raise ValueError("الملف الصوتي فارغ")
            
            chunks = []
            
            # إنشاء مجلد مؤقت للأجزاء
            temp_chunk_dir = TEMP_DIR / f"chunks_{int(time.time())}"
            temp_chunk_dir.mkdir(exist_ok=True)
            
            try:
                for i in range(0, int(duration), chunk_duration):
                    # التحقق من الموارد قبل معالجة كل جزء
                    if not ResourceManager.check_system_resources():
                        raise ResourceWarning("موارد النظام غير كافية لمواصلة المعالجة")
                    
                    start = i
                    end = min(i + chunk_duration, duration)
                    chunk = audio[start:end]
                    
                    if len(chunk) == 0:
                        continue
                    
                    # حفظ الجزء في ملف مؤقت
                    chunk_path = temp_chunk_dir / f"chunk_{i}.wav"
                    chunk.export(chunk_path, format="wav", codec="pcm_s16le")
                    
                    # التحقق من صحة الجزء
                    if not AudioProcessor.validate_audio_file(str(chunk_path)):
                        ErrorLogger.log_error("ChunkValidation", f"جزء غير صالح: chunk_{i}.wav")
                        continue
                    
                    # تحسين الجزء
                    optimized_path = temp_chunk_dir / f"chunk_{i}_opt.wav"
                    if AudioProcessor.optimize_audio_for_whisper(str(chunk_path), str(optimized_path)):
                        try:
                            # التعرف على الكلام في الجزء باستخدام Faster Whisper
                            segments, info = RetryManager.with_retry(
                                model.transcribe,
                                str(optimized_path),
                                beam_size=5,
                                word_timestamps=True,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=500)
                            )
                            chunk_text = " ".join([segment.text for segment in segments]).strip()
                            if chunk_text:
                                chunks.append(chunk_text)
                        except Exception as e:
                            ErrorLogger.log_error("ChunkProcessing", f"فشل معالجة الجزء {i}", str(e))
                    
                    # تنظيف الملفات المؤقتة للجزء
                    chunk_path.unlink(missing_ok=True)
                    optimized_path.unlink(missing_ok=True)
            
            finally:
                # تنظيف مجلد الأجزاء
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)
            
            if not chunks:
                raise ValueError("لم يتم التعرف على أي نص في جميع الأجزاء")
            
            transcription = " ".join(chunks)
            
            # حفظ النتيجة في التخزين المؤقت
            AudioCache.save_to_cache(audio_path, transcription)
            return transcription
            
        except Exception as e:
            ErrorLogger.log_error("LargeAudioProcessing", "فشل معالجة الملف الصوتي الكبير", str(e))
            raise RuntimeError(f"فشل معالجة الملف الصوتي الكبير: {str(e)}")


class GroqAPI:
    """فئة للتعامل مع Groq API"""
    
    _last_request_time = 0  # تتبع وقت آخر طلب
    
    @staticmethod
    def _wait_for_rate_limit():
        """الانتظار للامتثال لحدود المعدل"""
        current_time = time.time()
        time_since_last_request = current_time - GroqAPI._last_request_time
        if time_since_last_request < GROQ_RATE_LIMIT_DELAY:
            wait_time = GROQ_RATE_LIMIT_DELAY - time_since_last_request
            print(f"⏳ الانتظار {wait_time:.1f} ثوانٍ للامتثال لحد المعدل...")
            time.sleep(wait_time)
        GroqAPI._last_request_time = time.time()
    
    @staticmethod
    def _handle_rate_limit(retry_count):
        """التعامل مع أخطاء حد المعدل"""
        if retry_count >= GROQ_MAX_RETRIES:
            raise RuntimeError("تم تجاوز الحد الأقصى لمحاولات إعادة المحاولة")
        # زيادة التأخير بشكل تصاعدي مع حد أقصى
        wait_time = min(GROQ_RATE_LIMIT_DELAY * (2 ** retry_count), GROQ_MAX_BACKOFF)
        print(f"⏳ الانتظار {wait_time} ثوانٍ قبل إعادة المحاولة...")
        time.sleep(wait_time)
    
    @staticmethod
    def _make_api_request(data, retry_count=0):
        """إجراء طلب API مع إعادة المحاولة"""
        GroqAPI._wait_for_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
            print(f"📡 حالة الاستجابة: {response.status_code}")
            
            if response.status_code == 429:
                if retry_count < GROQ_MAX_RETRIES:
                    print(f"⚠ تم تجاوز حد المعدل (محاولة {retry_count + 1}/{GROQ_MAX_RETRIES})")
                    GroqAPI._handle_rate_limit(retry_count)
                    return GroqAPI._make_api_request(data, retry_count + 1)
                else:
                    raise RuntimeError("تم تجاوز الحد الأقصى لمحاولات إعادة المحاولة")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if retry_count < GROQ_MAX_RETRIES:
                print(f"⚠ خطأ في الطلب (محاولة {retry_count + 1}/{GROQ_MAX_RETRIES}): {str(e)}")
                GroqAPI._handle_rate_limit(retry_count)
                return GroqAPI._make_api_request(data, retry_count + 1)
            raise
    
    @staticmethod
    def test_connection():
        """اختبار الاتصال بـ Groq API"""
        try:
            data = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "user", "content": "Hello, this is a test message."}
                ]
            }
            
            print("🔄 اختبار الاتصال بـ Groq API...")
            print(f"🔗 URL: {GROQ_API_URL}")
            print(f"🤖 النموذج: {GROQ_MODEL}")
            print(f"📤 البيانات المرسلة: {data}")
            
            result = GroqAPI._make_api_request(data)
            print("✅ تم الاتصال بنجاح!")
            return True
        except Exception as e:
            print(f"❌ فشل الاتصال: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"📄 تفاصيل الخطأ: {e.response.text}")
            return False
    
    @staticmethod
    def get_summary(text, lang="en"):
        """الحصول على ملخص النص باستخدام Groq API"""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY غير متوفر في متغيرات البيئة")
        
        # تحديد System Prompt بناءً على اللغة
        meeting_prompt = (
            "You are an expert meeting summarizer.\n\n"
            "I will give you a transcript of a meeting. Please do the following:\n\n"
            "1. Summarize the key discussion points in a clear and concise way.\n"
            "2. Extract and list all decisions that were made during the meeting.\n"
            "3. If applicable, include any action items or tasks and who they were assigned to.\n\n"
            "Output format:\n"
            "- Summary:\n[Your summary here]\n\n"
            "- Decisions Made:\n[Bullet list of decisions]\n\n"
            "- Action Items:\n[Bullet list of action items, if mentioned]\n\n"
            "Meeting Transcript:\n\"\"\"\n[ضع هنا نص الاجتماع]\n\"\"\"\n"
        )
        system_prompt = meeting_prompt
        
        print(f"🔄 جاري الاتصال بـ Groq API...")
        print(f"📝 اللغة المستخدمة: {lang}")
        
        try:
            # تقسيم النص إلى أجزاء إذا كان طويلاً
            MAX_CHUNK_SIZE = 4000
            text_chunks = TextProcessor.fast_text_split(text, MAX_CHUNK_SIZE)
            summaries = []
            
            for i, chunk in enumerate(text_chunks, 1):
                print(f"🔄 معالجة الجزء {i} من {len(text_chunks)}...")
                
                data = {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ]
                }
                
                print(f"📤 إرسال الطلب للجزء {i}...")
                result = GroqAPI._make_api_request(data)
                
                if "choices" in result and len(result["choices"]) > 0:
                    chunk_summary = result["choices"][0]["message"]["content"].strip()
                    summaries.append(chunk_summary)
                    print(f"✅ تم استخراج ملخص الجزء {i} بنجاح")
                else:
                    print(f"❌ لم يتم العثور على ملخص في الرد للجزء {i}")
                    print(f"📄 محتوى الرد: {result}")
                    raise ValueError(f"لم يتم العثور على ملخص في استجابة Groq API للجزء {i}")
            
            # دمج الملخصات
            if len(summaries) > 1:
                combined_summary = " ".join(summaries)
                if len(combined_summary) > MAX_CHUNK_SIZE:
                    return GroqAPI.get_summary(combined_summary, lang)  # تلخيص تكراري
                return combined_summary
            elif len(summaries) == 1:
                return summaries[0]
            else:
                raise ValueError("لم يتم إنشاء أي ملخصات")
        except Exception as e:
            print(f"❌ خطأ غير متوقع: {str(e)}")
            ErrorLogger.log_error("GroqAPI", "خطأ في معالجة طلب Groq API", str(e))
            raise RuntimeError(f"خطأ في معالجة طلب Groq API: {str(e)}")

class GroqTranslator:
    """فئة للترجمة باستخدام Groq API"""
    
    def __init__(self):
        """تهيئة المتغيرات الأساسية"""
        self.api_key = GROQ_API_KEY
        if not self.api_key:
            raise ValueError("يجب تعيين GROQ_API_KEY في متغيرات البيئة")
        
        self.api_url = GROQ_API_URL
        self.model = GROQ_MODEL
        
        # إعدادات التأخير وإعادة المحاولة
        self.initial_delay = 3  # تأخير أولي 3 ثوانٍ
        self.max_retries = 5
        self.max_backoff = 30  # الحد الأقصى للتأخير 30 ثانية
    
    def _get_system_prompt(self, source_lang, target_lang):
        """تحديد التوجيه المناسب بناءً على اتجاه الترجمة"""
        if source_lang == 'ar' and target_lang == 'en':
            return """You are a professional Arabic to English translator. Translate the following Arabic text into fluent, academic English. Maintain the original meaning and context. Translate historical terms and names accurately. Do not add any explanations or notes. Keep the translation formal and precise. Only return the translation, nothing else."""
        elif source_lang == 'en' and target_lang == 'ar':
            return """أنت مترجم محترف من الإنجليزية إلى العربية. قم بترجمة النص التالي إلى اللغة العربية الفصحى بأسلوب أكاديمي سلس. حافظ على المعنى والسياق الأصلي. ترجم المصطلحات والأسماء التاريخية بدقة. لا تضف أي شروحات أو ملاحظات. حافظ على الأسلوب الرسمي والدقيق. قم بإرجاع الترجمة فقط، بدون أي إضافات."""
        else:
            raise ValueError(f"اتجاه الترجمة غير مدعوم: من {source_lang} إلى {target_lang}")
    
    def _make_api_request(self, text, system_prompt, retry_count=0):
        """إرسال طلب إلى Groq API مع دعم إعادة المحاولة"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.3,  # قيمة منخفضة للحصول على نتائج أكثر دقة
            "max_tokens": 4000
        }
        
        try:
            delay = min(self.initial_delay * (2 ** retry_count), self.max_backoff)
            time.sleep(delay)
            
            print(f"🔄 إرسال طلب الترجمة (محاولة {retry_count + 1})")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:  # تجاوز حد المعدل
                if retry_count < self.max_retries:
                    print(f"⚠ تم تجاوز حد المعدل. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                    return self._make_api_request(text, system_prompt, retry_count + 1)
                raise Exception("تم تجاوز الحد الأقصى لمحاولات إعادة المحاولة")
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                translation = result["choices"][0]["message"]["content"].strip()
                print("✅ تمت الترجمة بنجاح")
                return translation
            
            raise Exception("استجابة غير صالحة من API")
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                print(f"⚠ خطأ في الطلب. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                return self._make_api_request(text, system_prompt, retry_count + 1)
            raise Exception(f"فشل الاتصال بـ Groq API: {str(e)}")
        except Exception as e:
            if retry_count < self.max_retries:
                print(f"⚠ خطأ غير متوقع. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                time.sleep(self.initial_delay)
                return self._make_api_request(text, system_prompt, retry_count + 1)
            raise

    def split_text(self, text: str) -> List[str]:
        """تقسيم النص إلى أجزاء لا تتجاوز الحد المسموح من التوكنز"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > self.chunk_size:
                sub_sentences = sentence.split('،' if self.detect_language(sentence) == 'ar' else ',')
                for sub_sentence in sub_sentences:
                    sub_sentence = sub_sentence.strip() + ','
                    sub_tokens = self.count_tokens(sub_sentence)
                    
                    if current_length + sub_tokens > self.chunk_size:
                        chunks.append('.'.join(current_chunk))
                        current_chunk = [sub_sentence]
                        current_length = sub_tokens
                    else:
                        current_chunk.append(sub_sentence)
                        current_length += sub_tokens
            else:
                if current_length + sentence_tokens > self.chunk_size:
                    chunks.append('.'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('.'.join(current_chunk))
        
        return chunks
    
    def translate(self, text, target_lang=None, source_lang=None, max_chunk_size=4000):
        try:
            if not text or not text.strip():
                return text

            # تحديد اللغات المصدر والهدف
            source_lang = source_lang or detect(text[:1000])
            target_lang = target_lang or "en"

            # التحقق من الذاكرة المؤقتة
            cached = TranslationCache.get_cached_translation(text, source_lang, target_lang)
            if cached:
                print("✅ تم استرجاع الترجمة من الذاكرة المؤقتة")
                return cached

            # تحضير التوجيه للنموذج
            system_prompt = self._get_system_prompt(source_lang, target_lang)

            # تقسيم النص إذا كان طويلاً
            if len(text) > max_chunk_size:
                print("⚡ النص طويل، سيتم تقسيمه...")
                chunks = self.split_text(text)
                translated_chunks = []

                for i, chunk in enumerate(chunks, 1):
                    print(f"\n🔄 ترجمة الجزء {i}/{len(chunks)}...")
                    translated_chunk = self._make_api_request(chunk, system_prompt)
                    if translated_chunk:
                        translated_chunks.append(translated_chunk)
                        print(f"✓ تمت ترجمة الجزء {i}")
                    else:
                        raise Exception(f"فشل ترجمة الجزء {i}")

                    # تأخير بين الأجزاء
                    if i < len(chunks):
                        delay = min(2 * i, 5)  # تأخير تدريجي مع حد أقصى 5 ثوانٍ
                        print(f"⏳ انتظار {delay} ثوانٍ قبل الجزء التالي...")
                        time.sleep(delay)

                if translated_chunks:
                    final_translation = " ".join(translated_chunks)
                    # تخزين في الذاكرة المؤقتة
                    TranslationCache.cache_translation(text, source_lang, target_lang, final_translation)
                    print("✅ تمت الترجمة وحفظها في الذاكرة المؤقتة")
                    return final_translation
                else:
                    raise Exception("لم يتم ترجمة أي جزء بنجاح")
            else:
                print("ℹ النص قصير بما يكفي للترجمة مباشرة")
                translation = self._make_api_request(text, system_prompt)
                if translation:
                    # تخزين في الذاكرة المؤقتة
                    TranslationCache.cache_translation(text, source_lang, target_lang, translation)
                    print("✅ تمت الترجمة وحفظها في الذاكرة المؤقتة")
                    return translation
                else:
                    raise Exception("فشلت عملية الترجمة")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ خطأ في الترجمة: {error_msg}")
            ErrorLogger.log_error("Translation", "فشل في الترجمة", error_msg)
            return text

class SummaryTranslator:
    """فئة للتلخيص والترجمة"""
    
    _translator = None
    
    @staticmethod
    def get_translator():
        """الحصول على نسخة واحدة من المترجم"""
        if SummaryTranslator._translator is None:
            SummaryTranslator._translator = GroqTranslator()
        return SummaryTranslator._translator
    
    @staticmethod
    def process_text_for_summary(text, source_lang=None, target_lang="en"):
        """معالجة النص للتلخيص مباشرة باللغة المطلوبة مع تنظيف الملخص النهائي دومًا"""
        try:
            if not source_lang:
                source_lang = TextProcessor.detect_language_safe(text)
                source_lang = LanguageManager.normalize_language_code(source_lang)
            summarizer = GroqSummarizer()
            result = summarizer.summarize(text)
            if result["status"] == "success":
                summary = result["final_summary"]
                # إذا كانت اللغة المطلوبة مختلفة عن لغة المصدر
                if target_lang != source_lang:
                    translator = SummaryTranslator.get_translator()
                    translated_summary = translator.translate(
                        summary,
                        target_lang=target_lang,
                        source_lang=source_lang
                    )
                    if translated_summary:
                        # تنظيف الملخص النهائي بعد الترجمة
                        cleaned = summarizer.clean_summary(translated_summary)
                        return [cleaned, None]
                # إذا كانت اللغة المطلوبة هي نفس لغة المصدر
                cleaned = summarizer.clean_summary(summary)
                return [cleaned, None]
            else:
                # حتى في حالة الخطأ، نظف النص المرجع
                cleaned = summarizer.clean_summary(text)
                return [cleaned, result["message"]]
        except Exception as e:
            print(f"❌ خطأ في معالجة النص: {str(e)}")
            summarizer = GroqSummarizer()
            cleaned = summarizer.clean_summary(text)
            return [cleaned, str(e)]
    
    @staticmethod
    def translate_text(text, target_lang="en", source_lang=None):
        """ترجمة النص باستخدام Groq API"""
        translator = SummaryTranslator.get_translator()
        return translator.translate(text, target_lang=target_lang, source_lang=source_lang)

class GroqSummarizer:
    """فئة للتعامل مع تلخيص النصوص الطويلة باستخدام Groq API"""
    
    def __init__(self):
        """تهيئة المتغيرات الأساسية"""
        self.api_key = GROQ_API_KEY
        if not self.api_key:
            raise ValueError("يجب تعيين GROQ_API_KEY في متغيرات البيئة")
        
        self.api_url = GROQ_API_URL
        self.model = GROQ_MODEL
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # إعدادات التأخير وإعادة المحاولة
        self.initial_delay = GROQ_RATE_LIMIT_DELAY
        self.max_retries = GROQ_MAX_RETRIES
        self.max_backoff = GROQ_MAX_BACKOFF
        
        # حدود التوكنز
        self.max_tokens = 7000
        self.chunk_size = 6000
    
    def count_tokens(self, text: str) -> int:
        """حساب عدد التوكنز في النص"""
        return len(self.encoding.encode(text))
    
    def detect_language(self, text: str) -> str:
        """تحديد لغة النص"""
        try:
            return detect(text[:1000])
        except:
            return 'en'
    
    def get_system_prompt(self, lang: str) -> str:
        """تحديد التوجيه المناسب بناءً على لغة النص"""
        if lang == 'ar':
            return "قم بتلخيص النص التالي بلغة عربية واضحة ومختصرة، بدون إعادة كتابة هذه التعليمات في المخرجات:"
        return "Summarize the following text in clear and concise English, without including these instructions in the output:"
    
    def split_text(self, text: str) -> List[str]:
        """تقسيم النص إلى أجزاء لا تتجاوز الحد المسموح من التوكنز"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > self.chunk_size:
                sub_sentences = sentence.split('،' if self.detect_language(sentence) == 'ar' else ',')
                for sub_sentence in sub_sentences:
                    sub_sentence = sub_sentence.strip() + ','
                    sub_tokens = self.count_tokens(sub_sentence)
                    
                    if current_length + sub_tokens > self.chunk_size:
                        chunks.append('.'.join(current_chunk))
                        current_chunk = [sub_sentence]
                        current_length = sub_tokens
                    else:
                        current_chunk.append(sub_sentence)
                        current_length += sub_tokens
            else:
                if current_length + sentence_tokens > self.chunk_size:
                    chunks.append('.'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('.'.join(current_chunk))
        
        return chunks
    
    def make_api_request(self, text: str, system_prompt: str, retry_count: int = 0) -> str:
        """إرسال طلب إلى Groq API مع دعم إعادة المحاولة"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        }
        
        try:
            delay = min(self.initial_delay * (2 ** retry_count), self.max_backoff)
            time.sleep(delay)
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    print(f"⚠ تم تجاوز حد المعدل. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                    return self.make_api_request(text, system_prompt, retry_count + 1)
                raise Exception("تم تجاوز الحد الأقصى لمحاولات إعادة المحاولة")
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            raise Exception("استجابة غير صالحة من API")
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                print(f"⚠ خطأ في الطلب. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                return self.make_api_request(text, system_prompt, retry_count + 1)
            raise Exception(f"فشل الاتصال بـ Groq API: {str(e)}")
        except Exception as e:
            if retry_count < self.max_retries:
                print(f"⚠ خطأ غير متوقع. إعادة المحاولة {retry_count + 1}/{self.max_retries}")
                time.sleep(self.initial_delay)
                return self.make_api_request(text, system_prompt, retry_count + 1)
            raise
    
    def clean_summary(self, text: str) -> str:
        """تنظيف وتنسيق النص الملخص بشكل ذكي: إذا كان عربي يحذف الكلمات غير العربية، وإذا إنجليزي يحذف الكلمات غير الإنجليزية."""
        if not text or not isinstance(text, str):
            return ""
        # تنظيف أولي للرموز والمسافات
        text = re.sub(r'[\*\_\#\=\~\^\[\]\{\}\(\)\<\>\|\$\%\@\`\"\;]+', '', text)
        text = re.sub(r'[•●▪️✔️★☆→←↑↓※§¤°±×÷]', '', text)
        text = re.sub(r'\b(THEN|SHORT SUMMARY|SUMMARY|ملخص|ملخص:|خلاصة|خلاصة:|ملخصًا|ملخصا)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[A-Za-z]+:', '', text)
        text = re.sub(r'http\S+|www\S+|@[\w_]+', '', text)
        text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
        text = re.sub(r'[\u061f\u060c\u061b]', '.', text)
        text = re.sub(r'[؟?!،؛]+', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'[^\u0600-\u06FFa-zA-Z0-9\s\.\,\؟\!\-\n]', '', text)
        # تصحيح بعض الترجمات الشائعة
        text = text.replace('Devlet', 'الدولة')
        text = text.replace('Caliph', 'خليفة')
        text = text.replace('Omawi', 'أموي')
        text = text.replace('expanded', 'توسعت')
        text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)
        # تحديد اللغة الغالبة في النص
        arabic_count = len(re.findall(r'[\u0600-\u06FF]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        lang = 'ar' if arabic_count >= english_count else 'en'
        # تقسيم الجمل وتنظيفها
        sentences = [s.strip() for s in re.split(r'[\.\!؟]', text) if s.strip()]
        cleaned_sentences = []
        for s in sentences:
            words = s.split()
            filtered_words = []
            for w in words:
                # حذف الكلمات غير المتوافقة مع لغة النص
                if lang == 'ar':
                    if re.match(r'^[\u0600-\u06FF0-9]+$', w):
                        filtered_words.append(w)
                else:
                    if re.match(r'^[a-zA-Z0-9]+$', w):
                        filtered_words.append(w)
            # حذف الكلمات القصيرة جدًا (حرف أو حرفين) إلا إذا كانت رقم
            filtered_words = [w for w in filtered_words if len(w) > 2 or w.isdigit()]
            if filtered_words:
                new_s = ' '.join(filtered_words)
                if lang == 'en' and re.match(r'^[a-zA-Z]', new_s):
                    new_s = new_s.capitalize()
                cleaned_sentences.append(new_s)
        cleaned_text = '. '.join(cleaned_sentences) + '.' if cleaned_sentences else ''
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def summarize(self, text: str) -> Dict[str, Union[str, List[str], int]]:
        """تلخيص النص مع دعم التقسيم للنصوص الطويلة"""
        if not text.strip():
            raise ValueError("النص فارغ")
        
        # الكشف عن اللغة وتوحيد رمزها
        lang = LanguageManager.normalize_language_code(TextProcessor.detect_language_safe(text))
        if not LanguageManager.is_valid_language(lang):
            print(f"⚠ اللغة '{lang}' غير مدعومة، سيتم استخدام الإنجليزية")
            lang = "en"
        
        system_prompt = self.get_system_prompt(lang)
        total_tokens = self.count_tokens(text)
        print(f"\n📊 إجمالي التوكنز: {total_tokens}")
        
        try:
            # محاولة التلخيص باستخدام Groq API
            if total_tokens <= self.max_tokens:
                print("💡 النص قصير بما يكفي للتلخيص المباشر")
                summary = self.make_api_request(text, system_prompt)
                summary = self.clean_summary(summary)
                return {
                    "status": "success",
                    "original_tokens": total_tokens,
                    "chunks": [text],
                    "chunk_summaries": [summary],
                    "final_summary": summary,
                    "language": lang
                }
            
            # تقسيم النص الطويل إلى أجزاء
            chunks = TextChunker.split_text(text, self.chunk_size)
            print(f"📑 تم تقسيم النص إلى {len(chunks)} أجزاء")
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks, 1):
                print(f"\n🔄 جاري تلخيص الجزء {i}/{len(chunks)}")
                chunk_tokens = self.count_tokens(chunk)
                print(f"📝 عدد توكنز الجزء: {chunk_tokens}")
                
                try:
                    chunk_summary = self.make_api_request(chunk, system_prompt)
                    chunk_summary = self.clean_summary(chunk_summary)
                    chunk_summaries.append(chunk_summary)
                    print(f"✅ تم تلخيص الجزء {i}")
                except Exception as e:
                    print(f"⚠ فشل تلخيص الجزء {i}, استخدام خطة بديلة...")
                    # خطة بديلة: استخدام أول وآخر جملتين من الجزء
                    fallback_summary = self._create_fallback_summary(chunk)
                    chunk_summaries.append(fallback_summary)
                
                # تأخير ذكي بين الطلبات
                if i < len(chunks):
                    delay = min(3 * i, 10)  # زيادة التأخير تدريجياً مع حد أقصى
                    print(f"⏳ انتظار {delay} ثوانٍ قبل الجزء التالي...")
                    time.sleep(delay)
            
            print("\n🔄 جاري إنشاء التلخيص النهائي...")
            combined_summary = "\n\n".join(chunk_summaries)
            
            try:
                final_summary = self.make_api_request(combined_summary, system_prompt)
                final_summary = self.clean_summary(final_summary)
            except Exception as e:
                print("⚠ فشل إنشاء التلخيص النهائي، استخدام الملخصات المجزأة...")
                final_summary = self._create_fallback_summary(combined_summary)
            
            print("✨ اكتمل التلخيص النهائي")
            
            return {
                "status": "success",
                "original_tokens": total_tokens,
                "chunks": chunks,
                "chunk_summaries": chunk_summaries,
                "final_summary": final_summary,
                "language": lang
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ حدث خطأ: {error_msg}")
            ErrorLogger.log_error("Summarization", "فشل في التلخيص", error_msg)
            
            # محاولة إنشاء ملخص بسيط في حالة الفشل
            fallback_summary = self._create_fallback_summary(text)
            
            return {
                "status": "partial_success",
                "message": f"تم إنشاء ملخص بسيط بسبب خطأ: {error_msg}",
                "original_tokens": total_tokens,
                "final_summary": fallback_summary,
                "language": lang
            }
    
    def _create_fallback_summary(self, text: str, max_sentences=3) -> str:
        """إنشاء ملخص بسيط في حالة فشل التلخيص الرئيسي"""
        try:
            # تقسيم النص إلى جمل
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text
            
            # اختيار أول وآخر الجمل
            if len(sentences) <= max_sentences:
                return '. '.join(sentences) + '.'
            
            # أخذ أول جملتين وآخر جملة
            selected_sentences = sentences[:2] + [sentences[-1]]
            return '. '.join(selected_sentences) + '.'
            
        except Exception as e:
            print(f"⚠ فشل إنشاء الملخص البسيط: {str(e)}")
            return text[:1000] + "..."  # إرجاع أول 1000 حرف كحل أخير

class LanguageManager:
    """فئة لإدارة اللغات والتحقق من صحتها"""
    
    SUPPORTED_LANGUAGES = {
        "ar": "العربية",
        "en": "الإنجليزية",
        # يمكن إضافة المزيد من اللغات هنا
    }
    
    @staticmethod
    def is_valid_language(lang_code):
        """التحقق من صحة رمز اللغة"""
        return lang_code in LanguageManager.SUPPORTED_LANGUAGES
    
    @staticmethod
    def normalize_language_code(lang_code):
        """توحيد رمز اللغة"""
        if not lang_code:
            return "en"
        lang_code = lang_code.lower().strip()
        if lang_code in ["ar", "arabic", "العربية"]:
            return "ar"
        return "en"
    
    @staticmethod
    def get_language_name(lang_code):
        """الحصول على اسم اللغة"""
        lang_code = LanguageManager.normalize_language_code(lang_code)
        return LanguageManager.SUPPORTED_LANGUAGES.get(lang_code, "غير معروفة")

class TranslationCache:
    """فئة للتخزين المؤقت للترجمات"""
    
    _cache = {}
    _max_cache_size = 1000  # الحد الأقصى لعدد العناصر في الذاكرة المؤقتة
    
    @staticmethod
    def _generate_cache_key(text, source_lang, target_lang):
        """إنشاء مفتاح فريد للتخزين المؤقت"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{text_hash}:{source_lang}:{target_lang}"
    
    @staticmethod
    def get_cached_translation(text, source_lang, target_lang):
        """استرجاع الترجمة المخزنة مؤقتاً"""
        cache_key = TranslationCache._generate_cache_key(text, source_lang, target_lang)
        cached_item = TranslationCache._cache.get(cache_key)
        if cached_item:
            print("✨ تم استرجاع الترجمة من الذاكرة المؤقتة")
            return cached_item
        return None
    
    @staticmethod
    def cache_translation(text, source_lang, target_lang, translation):
        """تخزين الترجمة في الذاكرة المؤقتة"""
        if len(TranslationCache._cache) >= TranslationCache._max_cache_size:
            # حذف أقدم عنصر عند امتلاء الذاكرة المؤقتة
            TranslationCache._cache.pop(next(iter(TranslationCache._cache)))
        
        cache_key = TranslationCache._generate_cache_key(text, source_lang, target_lang)
        TranslationCache._cache[cache_key] = translation
        print("✨ تم تخزين الترجمة في الذاكرة المؤقتة")

class TextChunker:
    """فئة لتقسيم النصوص الطويلة"""
    
    @staticmethod
    def split_text(text, max_chunk_size=4000, overlap=100):
        """تقسيم النص إلى أجزاء مع تداخل لضمان السياق"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            
            if end < len(text):
                # البحث عن نهاية الجملة أو الفقرة
                for separator in ["\n\n", "\n", ". ", ".", " "]:
                    split_pos = text.rfind(separator, start, end)
                    if split_pos != -1:
                        end = split_pos + len(separator)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # التحرك للجزء التالي مع مراعاة التداخل
            start = max(start + max_chunk_size - overlap, end)
        
        return chunks


# ============== صفحة تسجيل الدخول ==============
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'اسم المستخدم أو كلمة المرور غير صحيحة'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# ============== حماية الصفحات ==============
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# عدّل جميع المسارات لتتطلب تسجيل الدخول
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
@login_required
def process():
    try:
        print("===> Start /process endpoint")
        file = request.files.get('file')
        if not file:
            print("===> No file received")
            return jsonify({"error": "لم يتم تقديم ملف"})
        print(f"===> File received: {file.filename}")
        # استخراج معلمات الترجمة من البيانات المرسلة
        target_lang = request.form.get('target_lang', 'en')
        file_path, error = FileProcessor.process_uploaded_file(file)
        if error:
            print(f"===> Error in process_uploaded_file: {error}")
            return jsonify({"error": error})
        print(f"===> File saved to: {file_path}")
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm']:
            print(f"===> Processing audio file: {file_path}")
            text = FileProcessor.process_audio_file(file_path)
        else:
            print(f"===> Unsupported file type: {file_ext}")
            return jsonify({"error": "نوع الملف غير مدعوم، فقط الملفات الصوتية مسموحة"})
        print(f"===> Extracted text: {text}")
        if not text:
            print("===> No text found in file")
            return jsonify({"error": "لم يتم العثور على نص في الملف"})
        # الكشف عن لغة النص
        source_lang = TextProcessor.detect_language_safe(text)
        source_lang = LanguageManager.normalize_language_code(source_lang)
        # التلخيص مباشرة باللغة المطلوبة
        summary = SummaryTranslator.process_text_for_summary(
            text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        if not summary[0]:
            print("===> Failed to summarize text")
            return jsonify({"error": "فشل تلخيص النص"})
        return jsonify({
            "original_text": text if text.strip() else "لا يوجد نص متاح",
            "summary": summary[0],
            "source_lang": source_lang,
            "target_lang": target_lang
        })
    except Exception as e:
        ErrorLogger.log_error("Processing", "فشل معالجة الطلب", str(e))
        print(f"===> Exception in /process: {str(e)}")
        return jsonify({"error": f"خطأ في معالجة الطلب: {str(e)}"})

@app.route('/summarize', methods=['POST'])
@login_required
def summarize_text():
    """مسار لتلخيص النصوص وترجمة الملخص اختيارياً"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "status": "error",
                "message": "يجب توفير النص المراد تلخيصه"
            })
        
        text = data['text'].strip()
        should_translate = data.get('translate', False)  # التحقق من طلب الترجمة
        target_lang = data.get('target_lang', 'en')  # اللغة المطلوبة
        
        if not text:
            return jsonify({
                "status": "error",
                "message": "النص فارغ"
            })
        
        print("\n=== بدء عملية التلخيص ===")
        print(f"⏱ وقت البدء: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 الترجمة مطلوبة: {should_translate}")
        print(f"🌐 اللغة المطلوبة: {target_lang}")
        
        # الكشف عن لغة النص
        source_lang = TextProcessor.detect_language_safe(text)
        source_lang = LanguageManager.normalize_language_code(source_lang)
        print(f"📝 لغة النص المصدر: {source_lang}")
        
        # التلخيص
        summarizer = GroqSummarizer()
        result = summarizer.summarize(text)
        
        if result["status"] == "success":
            print("\n✅ تم التلخيص بنجاح")
            summary = result["final_summary"]
            
            # إذا تم طلب الترجمة وكانت اللغة المطلوبة مختلفة عن لغة المصدر
            if should_translate and target_lang != source_lang:
                print(f"\n🔄 جاري ترجمة الملخص من {source_lang} إلى {target_lang}...")
                translator = GroqTranslator()
                translated_summary = translator.translate(
                    summary,
                    target_lang=target_lang,
                    source_lang=source_lang
                )
                if translated_summary:
                    summary = translated_summary
                    print("✅ تمت ترجمة الملخص بنجاح")
        
            processing_time = time.time() - start_time
            print(f"\n⏱ زمن المعالجة الكلي: {processing_time:.2f} ثانية")
            
            return jsonify({
                "status": "success",
                "original_text": text,
                "summary": summary,
                "source_lang": source_lang if not should_translate else target_lang,
                "processing_time": f"{processing_time:.2f} seconds"
            })
        
        print("\n❌ فشلت عملية التلخيص")
        return jsonify({
            "status": "error",
            "message": result.get("message", "فشل في عملية التلخيص"),
            "processing_time": f"{time.time() - start_time:.2f} seconds"
        })
        
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"خطأ في المعالجة: {str(e)}",
            "processing_time": f"{time.time() - start_time:.2f} seconds"
        })

# ============== تشغيل التطبيق ==============

if __name__ == '__main__':
    print("Starting app.py ...")
    # التحقق من وجود مفتاح Groq API
    if not GROQ_API_KEY:
        print("⚠ تحذير: GROQ_API_KEY غير متوفر في متغيرات البيئة")
    else:
        # اختبار الاتصال بـ Groq API
        if not GroqAPI.test_connection():
            print("⚠ تحذير: فشل الاتصال بـ Groq API")
    
    mem = psutil.virtual_memory()
    cpu_count = os.cpu_count() or 1
    print(f"⚡ الذاكرة المتاحة: {mem.available / (1024**3):.2f} GB")
    print(f"⚡ عدد أنوية المعالج: {cpu_count}")
    
    if mem.available < 2 * 1024**3:
        print("⚠ تحذير: الذاكرة المتاحة قليلة، قد يؤثر على الأداء")
    
    # Clean up files older than 24 hours
    ResourceManager.cleanup_temp_files(older_than_hours=24)
    
    port = int(os.environ.get("PORT", 3000))  # Use 3000 for Replit compatibility
    app.run(
        debug=False,
        threaded=True,
        processes=1,
        host='0.0.0.0',
        port=port,
    )

