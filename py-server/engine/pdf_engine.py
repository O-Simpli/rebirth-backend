"""
PDF Processing Engine - Core Coordinator

The PDFEngine is the central coordinator for all PDF operations. It manages
resources, orchestrates processors, and provides a unified interface for
PDF extraction and modification operations.

Usage:
    >>> from engine.pdf_engine import PDFEngine
    >>> from engine.config import EngineConfig
    >>> 
    >>> config = EngineConfig(enable_caching=True)
    >>> with PDFEngine('document.pdf', config=config) as engine:
    ...     pages = engine.get_page_count()
    ...     print(f"Document has {pages} pages")
"""

import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict
from contextlib import contextmanager

import pdfplumber
import pikepdf

from engine.config import EngineConfig, PageRange
from engine.base_processor import BaseProcessor, ProcessorRegistry
from utils.validation import PdfValidationError, comprehensive_pdf_validation

logger = logging.getLogger(__name__)


class PDFEngine:
    """
    Unified PDF processing engine with resource management and processor coordination.
    
    Uses composition pattern to coordinate specialized processors for text extraction,
    image extraction, and content modification.
    
    Example:
        >>> with PDFEngine('document.pdf') as engine:
        ...     total_pages = engine.get_page_count()
    """
    
    def __init__(self, file_path: str, config: Optional[EngineConfig] = None):
        """
        Initialize PDF engine with file path and optional configuration.
        
        Note: Document is not opened until entering context manager (__enter__).
        
        Args:
            file_path: Path to PDF file to process
            config: Engine configuration (uses defaults if None)
            
        Raises:
            FileNotFoundError: If file does not exist
            PdfValidationError: If configuration is invalid
        """
        self.file_path = file_path
        self.config = config or EngineConfig.default()
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Validate configuration
        if not self.config.validate():
            raise PdfValidationError("Invalid engine configuration")
        
        # Resource handles (initialized in __enter__)
        self._pdfplumber_doc = None
        self._pikepdf_doc = None
        self._is_open = False
        
        # Page caching
        self._page_cache: Dict[int, Any] = {}
        self._cache_enabled = self.config.enable_caching
        
        # Processor registry
        self._processors = ProcessorRegistry()
        
        # Metadata
        self._page_count: Optional[int] = None
        self._file_size_mb: Optional[float] = None
        
        logger.debug(f"PDFEngine initialized for: {Path(file_path).name}")
    
    def __enter__(self) -> 'PDFEngine':
        """
        Enter context manager - open PDF and initialize resources.
        
        Returns:
            Self for use in with-statement
            
        Raises:
            PdfValidationError: If PDF cannot be opened or is invalid
        """
        try:
            logger.info(f"Opening PDF: {self.file_path}")
            
            # Validate PDF file
            if self.config.validate_on_open:
                self._validate_pdf_file()
            
            # Open PDF with pdfplumber (for text extraction)
            self._pdfplumber_doc = pdfplumber.open(self.file_path)
            
            # Open PDF with pikepdf (for modification)
            self._pikepdf_doc = pikepdf.open(self.file_path)
            
            # Cache page count
            self._page_count = len(self._pdfplumber_doc.pages)
            
            # Calculate file size
            self._file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
            
            self._is_open = True
            
            # Initialize processors (will be added in Phase 2+)
            self._initialize_processors()
            
            logger.info(
                f"PDF opened successfully: {self._page_count} pages, "
                f"{self._file_size_mb:.2f} MB"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            self._cleanup_resources()
            raise PdfValidationError(f"Failed to open PDF: {str(e)}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager - clean up all resources.
        
        Resources are cleaned up even if an exception occurred.
        """
        logger.info("Closing PDF engine")
        self._cleanup_resources()
        
        if exc_type is not None:
            logger.error(f"Exception during engine operation: {exc_val}")
        
        # Don't suppress exceptions
        return False
    
    def _validate_pdf_file(self) -> None:
        """
        Validate PDF file before processing.
        
        Raises:
            PdfValidationError: If validation fails
        """
        # Check file size
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise PdfValidationError(
                f"PDF file too large: {file_size_mb:.2f} MB "
                f"(max: {self.config.max_file_size_mb} MB)"
            )
        
        # Use comprehensive validation from utils
        try:
            comprehensive_pdf_validation(self.file_path, self.config.max_file_size_mb)
        except Exception as e:
            raise PdfValidationError(f"PDF validation failed: {str(e)}")
    
    def _initialize_processors(self) -> None:
        """Initialize all enabled processors."""
        if self.config.enable_text_processor:
            from engine.text_processor import TextProcessor, TextProcessorOptions
            options_dict = self.config.text_processor_options or {}
            options = TextProcessorOptions(**options_dict)
            self._processors.register('text', TextProcessor(self, options))
        
        if self.config.enable_image_processor:
            from engine.image_processor import ImageProcessor, ImageProcessorOptions
            options_dict = self.config.image_processor_options or {}
            options = ImageProcessorOptions(**options_dict)
            self._processors.register('image', ImageProcessor(self, options))
        
        if self.config.enable_content_modifier:
            from engine.content_modifier import ContentModifier, ContentModifierOptions
            options_dict = self.config.content_modifier_options or {}
            options = ContentModifierOptions(**options_dict)
            self._processors.register('modifier', ContentModifier(self, options))
        
        if self.config.enable_vector_processor:
            from engine.vector_processor import VectorProcessor
            from engine.config import VectorProcessorOptions
            options_dict = self.config.vector_processor_options or {}
            options = VectorProcessorOptions(**options_dict)
            self._processors.register('vector', VectorProcessor(self, options))
        
        # Initialize all registered processors
        self._processors.initialize_all()
    
    def _cleanup_resources(self) -> None:
        """
        Clean up all resources (documents, processors, cache).
        
        This method is idempotent and safe to call multiple times.
        """
        # Clean up processors first
        if self._processors:
            self._processors.cleanup_all()
        
        # Close documents
        if self._pdfplumber_doc is not None:
            try:
                self._pdfplumber_doc.close()
            except Exception as e:
                logger.warning(f"Error closing pdfplumber document: {e}")
            finally:
                self._pdfplumber_doc = None
        
        if self._pikepdf_doc is not None:
            try:
                self._pikepdf_doc.close()
            except Exception as e:
                logger.warning(f"Error closing pikepdf document: {e}")
            finally:
                self._pikepdf_doc = None
        
        # Clear cache
        self._page_cache.clear()
        
        self._is_open = False
    
    # Public API - Document Information
    
    def get_page_count(self) -> int:
        """
        Get total number of pages in document.
        
        Returns:
            Number of pages
            
        Raises:
            RuntimeError: If engine not opened
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        return self._page_count
    
    def get_file_size_mb(self) -> float:
        """
        Get file size in megabytes.
        
        Returns:
            File size in MB
            
        Raises:
            RuntimeError: If engine not opened
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        return self._file_size_mb
    
    def validate_page_range(self, page_range: PageRange) -> bool:
        """
        Validate that page range is within document bounds.
        
        Args:
            page_range: PageRange to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        return page_range.validate(self._page_count)
    
    # Public API - Resource Access (for processors)
    
    @property
    def pdfplumber_document(self):
        """
        Access pdfplumber document (for processors).
        
        Returns:
            pdfplumber PDF object
            
        Raises:
            RuntimeError: If engine not opened
        """
        if not self._is_open or self._pdfplumber_doc is None:
            raise RuntimeError("Engine not opened - use within context manager")
        
        return self._pdfplumber_doc
    
    @property
    def pikepdf_document(self):
        """
        Access pikepdf document (for processors).
        
        Returns:
            pikepdf Pdf object
            
        Raises:
            RuntimeError: If engine not opened
        """
        if not self._is_open or self._pikepdf_doc is None:
            raise RuntimeError("Engine not opened - use within context manager")
        
        return self._pikepdf_doc
    
    # Public API - Page Caching
    
    def get_cached_page(self, page_num: int) -> Optional[Any]:
        """
        Get cached page data if available.
        
        Args:
            page_num: 1-based page number
            
        Returns:
            Cached page data or None
        """
        if not self._cache_enabled:
            return None
        
        return self._page_cache.get(page_num)
    
    def cache_page(self, page_num: int, page_data: Any) -> None:
        """
        Cache page data for future use.
        
        Args:
            page_num: 1-based page number
            page_data: Page data to cache
        """
        if not self._cache_enabled:
            return
        
        # Implement simple LRU-like behavior
        if len(self._page_cache) >= self.config.max_cache_pages:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._page_cache))
            del self._page_cache[oldest_key]
        
        self._page_cache[page_num] = page_data
    
    def clear_cache(self) -> None:
        """Clear all cached page data."""
        self._page_cache.clear()
    
    # Public API - Helper Methods for Processors
    
    def get_page_height(self, page_index: int) -> float:
        """
        Get page height for coordinate transformations.
        
        Args:
            page_index: 0-based page index
            
        Returns:
            Page height in points
            
        Raises:
            RuntimeError: If engine not opened
            IndexError: If page index out of bounds
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        if page_index < 0 or page_index >= self._page_count:
            raise IndexError(f"Page index {page_index} out of bounds (0-{self._page_count-1})")
        
        # Get from pikepdf MediaBox (height = top - bottom)
        page = self._pikepdf_doc.pages[page_index]
        mediabox = page.MediaBox
        height = float(mediabox[3] - mediabox[1])
        return height
    
    def get_page_mediabox_bottom(self, page_index: int) -> float:
        """
        Get MediaBox bottom Y-coordinate for coordinate transformations.
        
        pdfminer.six operates in raw PDF coordinate space where the origin (0,0)
        may be outside the MediaBox. This method returns the MediaBox bottom offset
        which must be subtracted from pdfminer Y-coordinates to align with pikepdf
        coordinates that are relative to the MediaBox.
        
        Args:
            page_index: 0-based page index
            
        Returns:
            MediaBox bottom Y-coordinate in points
            
        Raises:
            RuntimeError: If engine not opened
            IndexError: If page index out of bounds
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        if page_index < 0 or page_index >= self._page_count:
            raise IndexError(f"Page index {page_index} out of bounds (0-{self._page_count-1})")
        
        page = self._pikepdf_doc.pages[page_index]
        mediabox = page.MediaBox
        return float(mediabox[1])
    
    def get_page_obstacles(self, page_index: int):
        """
        Extract obstacles (lines and rects) from a page.
        
        Args:
            page_index: 0-based page index
            
        Returns:
            PdfObstacles with lines and rects
            
        Raises:
            RuntimeError: If engine not opened
            IndexError: If page index out of bounds
        """
        if not self._is_open:
            raise RuntimeError("Engine not opened - use within context manager")
        
        if page_index < 0 or page_index >= self._page_count:
            raise IndexError(f"Page index {page_index} out of bounds (0-{self._page_count-1})")
        
        from models.pdf_types import PdfObstacles, PdfLine, PdfRect
        
        # Get pdfplumber page
        plumber_page = self._pdfplumber_doc.pages[page_index]
        
        # Extract obstacles with filtering
        min_linewidth = 0.5
        filtered_lines = []
        for line in plumber_page.lines:
            if line.get('linewidth', 0) >= min_linewidth:
                filtered_lines.append(PdfLine(**line))
        
        filtered_rects = []
        for rect in plumber_page.rects:
            width = rect.get('width', 0)
            height = rect.get('height', 0)
            is_horizontal_line = (height < 2 and width > 15)
            is_vertical_line = (width < 2 and height > 15)
            
            if is_horizontal_line or is_vertical_line:
                line_obj = {
                    'x0': rect['x0'], 'y0': rect['top'], 
                    'x1': rect['x1'], 'y1': rect['bottom'],
                    'linewidth': max(width, height)
                }
                filtered_lines.append(PdfLine(**line_obj))
                continue
            
            # Check if rect is stroked or filled
            is_stroked = rect.get('stroke', False)
            fill_color = rect.get('non_stroking_color')
            is_filled = rect.get('fill', False) and fill_color not in [None, (1, 1, 1)]
            is_large_enough = width > 5 and height > 5
            
            if (is_stroked or is_filled) and is_large_enough:
                rect_obj = {
                    'x0': rect['x0'], 
                    'y0': rect['top'],
                    'x1': rect['x1'], 
                    'y1': rect['bottom'],
                    'linewidth': rect.get('linewidth', 0),
                    'stroke': rect.get('stroke'),
                    'fill': rect.get('fill')
                }
                filtered_rects.append(PdfRect(**rect_obj))
        
        logger.debug(f"Page {page_index + 1}: Extracted {len(filtered_lines)} lines and {len(filtered_rects)} rects")
        return PdfObstacles(lines=filtered_lines, rects=filtered_rects)
    
    # Public API - Processor Access
    
    @property
    def text_processor(self):
        """Access TextProcessor instance."""
        processor = self._processors.get('text')
        if processor is None:
            raise RuntimeError("TextProcessor not enabled or not yet initialized")
        return processor
    
    @property
    def image_processor(self):
        """Access ImageProcessor instance."""
        processor = self._processors.get('image')
        if processor is None:
            raise RuntimeError("ImageProcessor not enabled or not yet initialized")
        return processor
    
    @property
    def content_modifier(self):
        """Access ContentModifier instance."""
        processor = self._processors.get('modifier')
        if processor is None:
            raise RuntimeError("ContentModifier not enabled or not yet initialized")
        return processor
    
    @property
    def vector_processor(self):
        """Access VectorProcessor instance."""
        processor = self._processors.get('vector')
        if processor is None:
            raise RuntimeError("VectorProcessor not enabled or not yet initialized")
        return processor
    
    # Status and Debugging
    
    @property
    def is_open(self) -> bool:
        """Check if engine is currently open."""
        return self._is_open
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            'is_open': self._is_open,
            'file_path': self.file_path,
            'page_count': self._page_count,
            'file_size_mb': self._file_size_mb,
            'cache_enabled': self._cache_enabled,
            'cached_pages': len(self._page_cache),
            'processors': self._processors.processor_names,
            'config': self.config.to_dict()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "open" if self._is_open else "closed"
        pages = f"{self._page_count} pages" if self._page_count else "unknown pages"
        return f"PDFEngine({Path(self.file_path).name}, {status}, {pages})"