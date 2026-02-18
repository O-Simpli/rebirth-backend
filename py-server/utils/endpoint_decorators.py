"""
Decorators for FastAPI endpoint error handling and resource management.

This module provides decorators to handle common patterns in PDF processing endpoints,
such as file validation, temporary file management, and error handling.
"""

import os
import tempfile
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, Optional
from fastapi import UploadFile, HTTPException, Request

from utils.validation import (
    validate_file_content,
    PdfValidationError,
    ProcessingTimeoutError,
    MemoryLimitError,
    VALIDATION_CONSTANTS
)

logger = logging.getLogger(__name__)


def handle_pdf_processing(func: Callable) -> Callable:
    """
    Decorator to handle common PDF processing patterns:
    - File type validation
    - File content reading and validation
    - Temporary file creation and cleanup
    - Processing timeout management
    - Standardized error handling
    
    The decorated function must accept `request: Request` as a keyword argument.
    The decorator will store data in `request.state`:
    - `request.state.temp_file_path`: Path to the temporary PDF file
    - `request.state.file_content`: Raw bytes of the uploaded file
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the request object from kwargs
        request: Request = kwargs.get('request')
        if not request:
            raise HTTPException(
                status_code=500,
                detail="Endpoint decorated with handle_pdf_processing must accept 'request: Request'"
            )
        
        # Extract the file and processing_timeout from kwargs
        file: UploadFile = kwargs.get('file')
        if not file:
            raise HTTPException(
                status_code=400,
                detail="File parameter is required"
            )
        
        # Get timeout value (defaults to configured max if not provided)
        processing_timeout: Optional[int] = kwargs.get('processing_timeout')
        timeout_seconds = processing_timeout or VALIDATION_CONSTANTS['MAX_PROCESSING_TIME_SECONDS']
        
        # Step 1: Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Step 2: Read and validate file content
        try:
            content = await file.read()
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading uploaded file: {str(e)}"
            )
        
        # Step 3: Validate file content (size, format, etc.)
        is_valid_content, content_error = validate_file_content(
            content,
            max_size_mb=VALIDATION_CONSTANTS['MAX_FILE_SIZE_MB']
        )
        
        if not is_valid_content:
            logger.warning(f"File content validation failed for {file.filename}: {content_error}")
            raise HTTPException(
                status_code=400,
                detail=content_error
            )
        
        # Step 4: Create temporary file for processing
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(content)
            temp_file.flush()
            temp_file.close()  # Close handle to allow processing on Windows
            
            # Store temp_file_path and file_content in request state
            request.state.temp_file_path = temp_file.name
            request.state.file_content = content
            
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Processing timed out after {timeout_seconds}s for {file.filename}")
                raise HTTPException(
                    status_code=408,
                    detail=f"PDF processing timed out after {timeout_seconds} seconds."
                )
            except PdfValidationError as e:
                logger.warning(f"PDF validation failed for {file.filename}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF validation failed: {str(e)}"
                )
            except ProcessingTimeoutError as e:
                logger.error(f"Processing timeout for {file.filename}: {e}")
                raise HTTPException(
                    status_code=408,
                    detail=f"Processing timeout: {str(e)}"
                )
            except MemoryLimitError as e:
                logger.error(f"Memory limit exceeded for {file.filename}: {e}")
                raise HTTPException(
                    status_code=507,
                    detail=f"Memory limit exceeded: {str(e)}"
                )
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                logger.error(f"Unexpected error processing {file.filename}: {e}")
                logger.exception("Full exception details:")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error during PDF processing: {str(e)}"
                )
                
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file.name}: {e}")
    
    return wrapper