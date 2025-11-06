"""
Model loader for background initialization of NLP models.
Handles lazy loading and caching of models for fast subsequent runs.
"""

import threading
import time
import logging
from typing import Dict, Optional, Any, List
from enum import Enum
from datetime import datetime

try:
    import spacy
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
except ImportError:
    spacy = None
    KeyBERT = None
    SentenceTransformer = None


class ModelStatus(Enum):
    """Model initialization status."""
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelLoader:
    """Singleton class for managing NLP model initialization."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.status = ModelStatus.NOT_STARTED
        self.models = {}
        self.error = None
        self._init_thread = None
        self._progress = {
            'spacy': False,
            'keybert': False,
            'sentence_transformer': False
        }
        self._logs = []
        self._current_model = None
        self._start_time = None
    
    def initialize_models(self, background: bool = True):
        """Initialize all NLP models.
        
        Args:
            background: If True, initialize in background thread
        """
        if self.status == ModelStatus.READY:
            return
        
        if self.status == ModelStatus.LOADING:
            return
        
        if background:
            if self._init_thread is None or not self._init_thread.is_alive():
                self.status = ModelStatus.LOADING
                self._init_thread = threading.Thread(target=self._load_models, daemon=True)
                self._init_thread.start()
        else:
            self._load_models()
    
    def _log(self, message: str, level: str = "INFO"):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self._logs.append(log_entry)
        # Keep only last 100 logs
        if len(self._logs) > 100:
            self._logs = self._logs[-100:]
    
    def _load_models(self):
        """Load all models synchronously."""
        self._start_time = time.time()
        self._log("Starting model initialization...")
        
        try:
            # Load spaCy
            self._current_model = "spacy"
            if spacy:
                try:
                    self._log("Loading spaCy model (en_core_web_sm)...")
                    self._progress['spacy'] = False  # Mark as loading
                    # Check if model is installed
                    try:
                        import subprocess
                        result = subprocess.run(['python', '-m', 'spacy', 'info', 'en_core_web_sm'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode != 0:
                            self._log("⚠️ spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm", "WARNING")
                            self.models['spacy'] = None
                            self._progress['spacy'] = False
                        else:
                            self.models['spacy'] = spacy.load("en_core_web_sm")
                            self._progress['spacy'] = True
                            elapsed = time.time() - self._start_time
                            self._log(f"✅ spaCy loaded successfully ({elapsed:.1f}s)")
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        # Try to load directly, if it fails, we'll catch it
                        self.models['spacy'] = spacy.load("en_core_web_sm")
                        self._progress['spacy'] = True
                        elapsed = time.time() - self._start_time
                        self._log(f"✅ spaCy loaded successfully ({elapsed:.1f}s)")
                except OSError as e:
                    self._log(f"⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm", "WARNING")
                    self._log(f"   Error details: {str(e)[:100]}", "WARNING")
                    self.models['spacy'] = None
                    self._progress['spacy'] = False
                except Exception as e:
                    self._log(f"❌ Error loading spaCy: {e}", "ERROR")
                    self.models['spacy'] = None
                    self._progress['spacy'] = False
            else:
                self._log("spaCy not available (not installed)", "WARNING")
            
            # Load KeyBERT
            self._current_model = "keybert"
            if KeyBERT:
                try:
                    self._log("Loading KeyBERT model...")
                    self._progress['keybert'] = False  # Mark as loading
                    self.models['keybert'] = KeyBERT()
                    self._progress['keybert'] = True
                    elapsed = time.time() - self._start_time
                    self._log(f"✅ KeyBERT loaded successfully ({elapsed:.1f}s)")
                except Exception as e:
                    self._log(f"❌ Error loading KeyBERT: {e}", "ERROR")
                    self.models['keybert'] = None
                    self.error = str(e)
            else:
                self._log("KeyBERT not available (not installed)", "WARNING")
            
            # Load Sentence Transformer
            self._current_model = "sentence_transformer"
            if SentenceTransformer:
                try:
                    self._log("Loading Sentence-BERT model (all-MiniLM-L6-v2)... This may take a while...")
                    self._progress['sentence_transformer'] = False  # Mark as loading
                    self.models['sentence_transformer'] = SentenceTransformer("all-MiniLM-L6-v2")
                    self._progress['sentence_transformer'] = True
                    elapsed = time.time() - self._start_time
                    self._log(f"✅ Sentence-BERT loaded successfully ({elapsed:.1f}s)")
                except Exception as e:
                    self._log(f"❌ Error loading Sentence-BERT: {e}", "ERROR")
                    self.models['sentence_transformer'] = None
                    if not self.error:
                        self.error = str(e)
            else:
                self._log("Sentence-BERT not available (not installed)", "WARNING")
            
            self._current_model = None
            total_time = time.time() - self._start_time
            self.status = ModelStatus.READY
            self._log(f"🎉 All models initialized successfully! Total time: {total_time:.1f}s")
            
        except Exception as e:
            self._current_model = None
            self.status = ModelStatus.FAILED
            self.error = str(e)
            self._log(f"❌ Critical error during model initialization: {e}", "ERROR")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model initialization status.
        
        Returns:
            Dict with status, progress, and models
        """
        return {
            'status': self.status.value,
            'progress': self._progress.copy(),
            'models_loaded': len([m for m in self.models.values() if m is not None]),
            'total_models': 3,
            'error': self.error,
            'current_model': self._current_model,
            'logs': self._logs.copy()
        }
    
    def get_models(self) -> Dict[str, Any]:
        """Get initialized models.
        
        Returns:
            Dict of model names to model objects
        """
        if self.status != ModelStatus.READY:
            return {}
        return self.models.copy()
    
    def is_ready(self) -> bool:
        """Check if models are ready."""
        return self.status == ModelStatus.READY
    
    def get_progress_percentage(self) -> float:
        """Get initialization progress as percentage.
        
        Returns:
            Float between 0.0 and 1.0
        """
        if self.status == ModelStatus.READY:
            return 1.0
        if self.status == ModelStatus.FAILED:
            return 0.0
        
        loaded = sum(1 for v in self._progress.values() if v)
        return loaded / 3.0

