"""
PDF File Validation and Resource Management Utilities
Comprehensive validation, resource monitoring, and error handling for PDF processing.
"""

import os
import tempfile
import time
import psutil
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Validation constants
VALIDATION_CONSTANTS = {
    'PDF_SIGNATURE': b'%PDF',
    'MAX_FILE_SIZE_MB': 50,
    'MAX_PROCESSING_TIME_SECONDS': 300,  # 5 minutes
    'MAX_MEMORY_USAGE_MB': 1000,  # 1GB
    'SUPPORTED_PDF_VERSIONS': ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '2.0'],
}

class PdfValidationError(Exception):
    """Custom exception for PDF validation errors"""
    pass

class ProcessingTimeoutError(Exception):
    """Custom exception for processing timeouts"""
    pass

class MemoryLimitError(Exception):
    """Custom exception for memory limit exceeded"""
    pass

def validate_pdf_signature(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate PDF file signature (magic bytes) and version
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            # Read first 8 bytes to check signature and version
            header = f.read(8)
            
            if len(header) < 4:
                return False, "File too small to be a valid PDF"
            
            # Check PDF signature
            if not header.startswith(VALIDATION_CONSTANTS['PDF_SIGNATURE']):
                return False, f"Invalid PDF signature. Expected {VALIDATION_CONSTANTS['PDF_SIGNATURE']}, got {header[:4]}"
            
            # Extract and validate PDF version
            if len(header) >= 8:
                try:
                    version_str = header[5:8].decode('ascii')
                    if version_str not in VALIDATION_CONSTANTS['SUPPORTED_PDF_VERSIONS']:
                        logger.warning(f"Unsupported PDF version: {version_str}")
                        # Continue processing - many PDFs work even with unsupported versions
                except UnicodeDecodeError:
                    logger.warning("Could not decode PDF version")
            
            return True, None
            
    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except PermissionError:
        return False, f"Permission denied accessing file: {file_path}"
    except Exception as e:
        return False, f"Error validating PDF signature: {str(e)}"

def validate_file_size(file_path: str, max_size_mb: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate file size limits
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB (uses default if None)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_size_mb is None:
        max_size_mb = VALIDATION_CONSTANTS['MAX_FILE_SIZE_MB']
    
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        
        logger.debug(f"File size validation passed: {size_mb:.1f}MB")
        return True, None
        
    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except Exception as e:
        return False, f"Error checking file size: {str(e)}"

def validate_file_content(content: bytes, max_size_mb: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file content before saving to disk
    
    Args:
        content: Raw file content bytes
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_size_mb is None:
        max_size_mb = VALIDATION_CONSTANTS['MAX_FILE_SIZE_MB']
    
    # Check size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Check PDF signature
    if len(content) < 4:
        return False, "File too small to be a valid PDF"
    
    if not content.startswith(VALIDATION_CONSTANTS['PDF_SIGNATURE']):
        return False, f"Invalid PDF signature in uploaded content"
    
    return True, None

def validate_processing_environment() -> Tuple[bool, Optional[str]]:
    """
    Validate that the system has sufficient resources for PDF processing
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        if available_mb < 100:  # Require at least 100MB available
            return False, f"Insufficient memory available: {available_mb:.1f}MB (need at least 100MB)"
        
        # Check disk space for temporary files
        temp_dir = tempfile.gettempdir()
        disk_usage = psutil.disk_usage(temp_dir)
        free_mb = disk_usage.free / (1024 * 1024)
        
        if free_mb < 100:  # Require at least 100MB free disk space
            return False, f"Insufficient disk space in {temp_dir}: {free_mb:.1f}MB (need at least 100MB)"
        
        logger.debug(f"Environment validation passed: {available_mb:.1f}MB memory, {free_mb:.1f}MB disk")
        return True, None
        
    except Exception as e:
        return False, f"Error checking system resources: {str(e)}"

class ResourceManager:
    """
    Context manager for tracking and limiting resource usage during PDF processing
    """
    
    def __init__(self, max_memory_mb: Optional[int] = None, max_time_seconds: Optional[int] = None):
        self.max_memory_mb = max_memory_mb or VALIDATION_CONSTANTS['MAX_MEMORY_USAGE_MB']
        self.max_time_seconds = max_time_seconds or VALIDATION_CONSTANTS['MAX_PROCESSING_TIME_SECONDS']
        self.start_time = None
        self.start_memory = None
        self.temp_files = []
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.debug(f"ResourceManager: Starting processing with {self.start_memory:.1f}MB memory")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
        
        # Log resource usage
        if self.start_time:
            processing_time = time.time() - self.start_time
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_delta = current_memory - self.start_memory if self.start_memory else 0
            
            logger.info(f"ResourceManager: Processing completed in {processing_time:.2f}s, "
                       f"memory usage: {memory_delta:+.1f}MB")
    
    def add_temp_file(self, file_path: str):
        """Register a temporary file for cleanup"""
        self.temp_files.append(file_path)
    
    def check_limits(self):
        """Check if resource limits have been exceeded"""
        current_time = time.time()
        
        # Check time limit
        if self.start_time and (current_time - self.start_time) > self.max_time_seconds:
            raise ProcessingTimeoutError(
                f"Processing timeout: {current_time - self.start_time:.1f}s "
                f"(max: {self.max_time_seconds}s)"
            )
        
        # Check memory limit
        try:
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_memory > self.max_memory_mb:
                raise MemoryLimitError(
                    f"Memory limit exceeded: {current_memory:.1f}MB "
                    f"(max: {self.max_memory_mb}MB)"
                )
        except Exception as e:
            logger.warning(f"Could not check memory usage: {e}")

def comprehensive_pdf_validation(file_path: str, max_size_mb: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform comprehensive PDF file validation
    
    Args:
        file_path: Path to the PDF file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    try:
        # File existence check
        if not os.path.exists(file_path):
            results['is_valid'] = False
            results['errors'].append(f"File not found: {file_path}")
            return results
        
        # File size validation
        size_valid, size_error = validate_file_size(file_path, max_size_mb)
        if not size_valid:
            results['is_valid'] = False
            results['errors'].append(size_error)
        else:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            results['file_info']['size_mb'] = round(size_mb, 2)
        
        # PDF signature validation
        sig_valid, sig_error = validate_pdf_signature(file_path)
        if not sig_valid:
            results['is_valid'] = False
            results['errors'].append(sig_error)
        
        # Environment validation
        env_valid, env_error = validate_processing_environment()
        if not env_valid:
            results['is_valid'] = False
            results['errors'].append(env_error)
        
        # Additional file info
        try:
            stat = os.stat(file_path)
            results['file_info'].update({
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'permissions': oct(stat.st_mode)[-3:]
            })
        except Exception as e:
            results['warnings'].append(f"Could not get file metadata: {e}")
        
    except Exception as e:
        results['is_valid'] = False
        results['errors'].append(f"Validation error: {str(e)}")
    
    return results

# Export main validation function for easy import
__all__ = [
    'validate_pdf_signature',
    'validate_file_size', 
    'validate_file_content',
    'validate_processing_environment',
    'comprehensive_pdf_validation',
    'ResourceManager',
    'PdfValidationError',
    'ProcessingTimeoutError',
    'MemoryLimitError',
    'VALIDATION_CONSTANTS'
]