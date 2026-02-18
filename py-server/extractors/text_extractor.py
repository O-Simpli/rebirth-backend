"""
PDF Content Extractor

High-performance PDF text extraction with character-level styling analysis
and intelligent text grouping. Optimized for accuracy and performance.

Uses PDFEngine + TextProcessor architecture for all extraction operations.
"""

import logging
from typing import Dict, List, Optional

from models.pdf_types import PdfTextElement 
from utils.validation import PdfValidationError

DEFAULT_START_PAGE = 1

logger = logging.getLogger(__name__)

def extract_text(
    file_path: str,
    start_page: int = DEFAULT_START_PAGE,
    end_page: Optional[int] = None,
    text_config: Optional[Dict] = None
) -> List[List[PdfTextElement]]:
    """Extract PDF content using two-pass architecture for horizontal and rotated text."""
    return _extract_text_new_engine(file_path, start_page, end_page, text_config)


def extract_text_with_stream_order(
    file_path: str,
    start_page: int = DEFAULT_START_PAGE,
    end_page: Optional[int] = None,
    text_config: Optional[Dict] = None
) -> List[List[PdfTextElement]]:
    """Extract PDF content in stream order with Z-index preservation."""
    return _extract_text_stream_order_new_engine(file_path, start_page, end_page, text_config)


# ==============================================================================
# Implementation using PDFEngine + TextProcessor
# ==============================================================================

def _extract_text_new_engine(
    file_path: str,
    start_page: int,
    end_page: Optional[int],
    text_config: Optional[Dict]
) -> List[List[PdfTextElement]]:
    """Extract text using new PDFEngine + TextProcessor"""
    try:
        from engine import PDFEngine, EngineConfig, TextProcessorOptions
        from processors.position_tracking import RegularTextStrategy
        
        # Configure engine
        engine_config = EngineConfig(
            enable_text_processor=True,
            enable_caching=True
        )
        
        # Map simplified API config to full engine config with smart defaults
        api_config = text_config or {}
        full_grouping_config = {
            'enable_grouping': api_config.get('enable_grouping', True),
            'enable_list_detection': api_config.get('enable_list_detection', True),
            'enable_toc_detection': api_config.get('enable_toc_detection', True),
            # Smart defaults for internal parameters
            'max_horizontal_gap': 5.0,
            'max_line_height_difference': 2.0,
            'max_vertical_gap': 30.0,
            'require_same_font': False,
            'require_similar_font_size': True,
            'font_size_tolerance': 2.0,
            'enable_semantic_grouping': False,
            'min_words_for_paragraph': 3,
            'max_list_item_gap': 50.0,
            'list_indent_tolerance': 15.0,
            'min_list_items': 2,
            'max_toc_horizontal_gap': 150.0,
            'min_toc_entries': 3,
            'toc_page_number_pattern': r'^\d+$|^[ivxlcdm]+$',
            'max_elements_to_consider': 10000,
        }
        
        # Configure text processor
        text_processor_options = TextProcessorOptions(
            enable_rotated_text=True,
            enable_text_grouping=full_grouping_config['enable_grouping'],
            text_grouping_config=full_grouping_config
        )
        
        with PDFEngine(file_path, config=engine_config) as engine:
            # Create text processor with custom options
            from engine.text_processor import TextProcessor
            text_processor = TextProcessor(engine, options=text_processor_options)
            text_processor.initialize()
            
            total_pages = engine.get_page_count()
            start_page = max(DEFAULT_START_PAGE, start_page)
            end_page = min(end_page, total_pages) if end_page is not None else total_pages
            
            if start_page > end_page:
                end_page = start_page

            logger.info(f"Processing PDF (new engine): {total_pages} total pages, extracting pages {start_page}-{end_page}")

            pages_data: List[List[PdfTextElement]] = []
            for page_num in range(start_page, end_page + 1):
                page_index = page_num - 1
                logger.debug(f"Processing page {page_num}")
                
                # Extract horizontal text using stream order device
                horizontal_elements = text_processor.extract_stream_elements(
                    page_index,
                    position_strategy=RegularTextStrategy()
                )
                
                # Extract rotated text
                rotated_runs = text_processor.extract_rotated_text(page_index)
                
                # Filter to text runs only (remove markers and images for this mode)
                from models.pdf_types import ProcessingTextRun
                horizontal_runs = [e for e in horizontal_elements if isinstance(e, ProcessingTextRun)]
                
                # Combine and sort by position
                all_runs = horizontal_runs + rotated_runs
                all_runs.sort(key=lambda run: (run.y, run.x))
                
                # Group content
                page_elements = text_processor.group_content(all_runs, text_config, page_num)
                pages_data.append(page_elements)
            
            text_processor.cleanup()
            logger.info(f"Extraction complete (new engine): {len(pages_data)} pages processed successfully")
            return pages_data

    except Exception as e:
        logger.error(f"PDF extraction failed (new engine): {e}", exc_info=True)
        raise PdfValidationError(f"PDF extraction failed: {str(e)}")


def _extract_text_stream_order_new_engine(
    file_path: str,
    start_page: int,
    end_page: Optional[int],
    text_config: Optional[Dict]
) -> List[List[PdfTextElement]]:
    """Extract text in stream order using new PDFEngine + TextProcessor"""
    try:
        from engine import PDFEngine, EngineConfig, TextProcessorOptions
        
        # Configure engine
        engine_config = EngineConfig(
            enable_text_processor=True,
            enable_caching=True
        )
        
        # Map simplified API config to full engine config with smart defaults
        api_config = text_config or {}
        full_grouping_config = {
            'enable_grouping': api_config.get('enable_grouping', True),
            'enable_list_detection': api_config.get('enable_list_detection', True),
            'enable_toc_detection': api_config.get('enable_toc_detection', True),
            # Smart defaults for internal parameters
            'max_horizontal_gap': 5.0,
            'max_line_height_difference': 2.0,
            'max_vertical_gap': 30.0,
            'require_same_font': False,
            'require_similar_font_size': True,
            'font_size_tolerance': 2.0,
            'enable_semantic_grouping': False,
            'min_words_for_paragraph': 3,
            'max_list_item_gap': 50.0,
            'list_indent_tolerance': 15.0,
            'min_list_items': 2,
            'max_toc_horizontal_gap': 150.0,
            'min_toc_entries': 3,
            'toc_page_number_pattern': r'^\\d+$|^[ivxlcdm]+$',
            'max_elements_to_consider': 10000,
        }
        
        # Configure text processor
        text_processor_options = TextProcessorOptions(
            enable_rotated_text=True,
            enable_text_grouping=full_grouping_config['enable_grouping'],
            text_grouping_config=full_grouping_config
        )
        
        with PDFEngine(file_path, config=engine_config) as engine:
            # Create text processor with custom options
            from engine.text_processor import TextProcessor
            text_processor = TextProcessor(engine, options=text_processor_options)
            text_processor.initialize()
            
            # Extract skip flags from config
            skip_text = text_config.get('skip_text_placeholders', False) if text_config else False
            skip_images = text_config.get('skip_image_placeholders', False) if text_config else False
            skip_vectors = text_config.get('skip_vector_placeholders', False) if text_config else False
            
            total_pages = engine.get_page_count()
            start_page = max(DEFAULT_START_PAGE, start_page)
            end_page = min(end_page, total_pages) if end_page is not None else total_pages
            
            if start_page > end_page:
                end_page = start_page

            logger.info(f"Processing PDF (stream order, new engine): {total_pages} total pages, extracting pages {start_page}-{end_page}")

            pages_data: List[List[PdfTextElement]] = []
            for page_num in range(start_page, end_page + 1):
                page_index = page_num - 1
                logger.debug(f"Processing page {page_num} (stream order)")
                
                # Get complete page stream (with rotated text integrated)
                stream_elements = text_processor.get_page_stream(
                    page_index,
                    skip_text_placeholders=skip_text,
                    skip_image_placeholders=skip_images,
                    skip_vector_placeholders=skip_vectors
                )
                
                # Group content preserving stream order
                page_elements = text_processor.group_content(stream_elements, text_config, page_num)
                pages_data.append(page_elements)
            
            text_processor.cleanup()
            logger.info(f"Stream order extraction complete (new engine): {len(pages_data)} pages processed successfully")
            return pages_data

    except Exception as e:
        logger.error(f"PDF stream extraction failed (new engine): {e}", exc_info=True)
        raise PdfValidationError(f"PDF extraction failed: {str(e)}")