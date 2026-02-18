"""
PDF Content Modification Module

This module provides a public API for removing content from PDFs.
Uses the new PDFEngine + ContentModifier architecture.
"""

import io
import logging
from typing import List, Dict
from pikepdf import Pdf

logger = logging.getLogger(__name__)


def remove_content_from_pdf(
        file_path: str,
        page_removals: dict
) -> bytes:
    """
    Remove text, images, and vectors from PDF pages.

    Args:
        file_path: Path to the input PDF file.
        page_removals: Dict mapping page numbers (1-based) to RemovalRequest objects
                      containing text_bboxes, image_bboxes, and vector_bboxes.

    Returns:
        Modified PDF as bytes.
    """
    return _remove_content_new_engine(file_path, page_removals)

def _remove_content_new_engine(file_path: str, page_removals: dict) -> bytes:
    """
    Remove content using new engine architecture (Phase 4).
    
    Uses ContentModifier processor for clean, composable implementation.
    """
    from engine import PDFEngine, EngineConfig
    
    if not page_removals:
        logger.warning("No page removals specified, returning original PDF")
        with open(file_path, 'rb') as f:
            return f.read()
    
    logger.info(f"Starting multi-page content removal on {len(page_removals)} page(s) using new engine")
    
    try:
        # Configure engine with only ContentModifier enabled
        config = EngineConfig(
            enable_text_processor=False,
            enable_image_processor=False,
            enable_content_modifier=True
        )
        
        with PDFEngine(file_path, config=config) as engine:
            modifier = engine.content_modifier
            total_pages = engine.get_page_count()
            
            logger.info(f"Processing PDF with {total_pages} total pages")
            
            # Validate all page numbers before processing
            for page_num in page_removals.keys():
                if page_num < 1 or page_num > total_pages:
                    raise ValueError(f"Page {page_num} is out of range. PDF has {total_pages} pages.")
            
            # Convert 1-based page numbers to 0-based indices
            page_removals_indexed = {}
            for page_num, removal_request in page_removals.items():
                page_index = page_num - 1
                
                # Extract removal data from the request
                text_bboxes = removal_request.text_bboxes or []
                image_bboxes = removal_request.image_bboxes or []
                vector_bboxes = removal_request.vector_bboxes or []
                
                logger.info(f"Queueing page {page_num}: {len(text_bboxes)} text regions, "
                           f"{len(image_bboxes)} images, {len(vector_bboxes)} vectors")
                
                page_removals_indexed[page_index] = {
                    'text_bboxes': text_bboxes,
                    'image_bboxes': image_bboxes,
                    'vector_bboxes': vector_bboxes
                }
            
            # Process all pages
            modifier.remove_content_from_pages(page_removals_indexed)
            
            # Get result
            result_bytes = modifier.get_modified_pdf_bytes()
            
            logger.info(f"Successfully processed {len(page_removals)} page(s), output size: {len(result_bytes)} bytes")
            return result_bytes
    
    except Exception as e:
        logger.error(f"Failed to process PDF with new engine: {e}")
        raise

