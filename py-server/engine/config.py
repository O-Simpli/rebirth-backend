"""
Configuration system for PDF Engine.

Provides structured configuration using dataclasses with clear defaults,
type safety, and backward compatibility with dict-based configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessorOptions:
    """
    Base class for processor-specific configuration options.
    
    All processor option classes should inherit from this to provide
    consistent interface and common functionality.
    """
    enabled: bool = True
    timeout_seconds: Optional[int] = None  # Override engine timeout if set
    
    def validate(self) -> bool:
        """
        Validate configuration options.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            logger.error("timeout_seconds must be non-negative")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'timeout_seconds': self.timeout_seconds
        }


@dataclass
class VectorProcessorOptions(ProcessorOptions):
    """
    Configuration options for vector graphics extraction.
    
    Controls behavior of gradient simplification, clip handling,
    and other vector-specific features.
    """
    # Extraction options
    extract_gradients: bool = True  # Extract gradient patterns
    extract_form_xobjects: bool = True  # Extract from Form XObjects
    simplify_paths: bool = False  # Simplify complex paths
    
    # Color handling
    preserve_color_spaces: bool = True  # Keep original color spaces
    
    # Output options
    generate_svg: bool = True  # Generate SVG content
    generate_path_data: bool = True  # Extract path data for rendering
    include_pattern_image_data: bool = False  # Include base64 image data for images embedded in patterns
    
    # Sparse vector splitting (content-aware bounds)
    split_sparse_vectors: bool = False  # Split large sparse vectors into tighter bounds
    sparse_vector_coverage_threshold: float = 20.0  # Minimum page coverage % to analyze (0-100)
    sparse_vector_merge_distance: float = 10.0  # Distance threshold in points for merging segments

    enable_vector_grouping: bool = False  # Group vectors by clipping region (W/W* operators)
    
    # Overlap merging (stream-based)
    enable_overlap_grouping: bool = False  # Name kept for compatibility, but now merges
    overlap_check_window: int = 20  # Last N vectors to check
    overlap_method: str = "area"  # "area", "iou", or "intersection"
    overlap_threshold: float = 0.1  # 10% by default
    proximity_threshold: Optional[float] = None  # Optional: merge by distance
    enable_post_processing_merge: bool = False  # Full O(N²) post-processing to catch all overlaps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'extract_gradients': self.extract_gradients,
            'extract_form_xobjects': self.extract_form_xobjects,
            'simplify_paths': self.simplify_paths,
            'preserve_color_spaces': self.preserve_color_spaces,
            'generate_svg': self.generate_svg,
            'generate_path_data': self.generate_path_data,
            'include_pattern_image_data': self.include_pattern_image_data,
            'split_sparse_vectors': self.split_sparse_vectors,
            'sparse_vector_coverage_threshold': self.sparse_vector_coverage_threshold,
            'sparse_vector_merge_distance': self.sparse_vector_merge_distance,
            'enable_vector_grouping': self.enable_vector_grouping,
            'enable_overlap_grouping': self.enable_overlap_grouping,
            'overlap_check_window': self.overlap_check_window,
            'overlap_method': self.overlap_method,
            'overlap_threshold': self.overlap_threshold,
            'proximity_threshold': self.proximity_threshold,
            'enable_post_processing_merge': self.enable_post_processing_merge,
        })
        return base_dict


@dataclass
class EngineConfig:
    """
    Central configuration for PDFEngine initialization.
    
    Provides all configuration options for engine behavior, resource management,
    and processor enablement.
    
    Example:
        >>> config = EngineConfig(enable_caching=True, max_cache_pages=20)
        >>> engine = PDFEngine(file_path, config=config)
    """
    
    # Resource management
    enable_caching: bool = True
    max_cache_pages: int = 10
    
    # Processing options
    enable_text_processor: bool = True
    enable_image_processor: bool = True
    enable_content_modifier: bool = True
    enable_vector_processor: bool = True
    
    # Processor-specific options (as dictionaries for flexibility)
    text_processor_options: Optional[Dict[str, Any]] = None
    image_processor_options: Optional[Dict[str, Any]] = None
    content_modifier_options: Optional[Dict[str, Any]] = None
    vector_processor_options: Optional[Dict[str, Any]] = None
    
    # Performance
    timeout_seconds: int = 300
    max_file_size_mb: int = 50
    
    # Validation
    validate_on_open: bool = True
    strict_mode: bool = False
    
    # Logging
    log_level: str = "INFO"
    enable_debug_logging: bool = False
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if all values are valid, False otherwise
        """
        if self.max_cache_pages < 0:
            logger.error("max_cache_pages must be non-negative")
            return False
        
        if self.timeout_seconds < 30:
            logger.error("timeout_seconds must be at least 30 seconds")
            return False
        
        if self.max_file_size_mb < 1:
            logger.error("max_file_size_mb must be at least 1 MB")
            return False
        
        if not any([self.enable_text_processor, 
                   self.enable_image_processor, 
                   self.enable_content_modifier,
                   self.enable_vector_processor]):
            logger.error("At least one processor must be enabled")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Useful for serialization, logging, and debugging.
        """
        return {
            'enable_caching': self.enable_caching,
            'max_cache_pages': self.max_cache_pages,
            'enable_text_processor': self.enable_text_processor,
            'enable_image_processor': self.enable_image_processor,
            'enable_content_modifier': self.enable_content_modifier,
            'enable_vector_processor': self.enable_vector_processor,
            'timeout_seconds': self.timeout_seconds,
            'max_file_size_mb': self.max_file_size_mb,
            'validate_on_open': self.validate_on_open,
            'strict_mode': self.strict_mode,
            'log_level': self.log_level,
            'enable_debug_logging': self.enable_debug_logging
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EngineConfig':
        """
        Create EngineConfig from dictionary.
        
        Provides backward compatibility with dict-based configuration.
        Unknown keys are ignored with a warning.
        
        Args:
            config: Dictionary of configuration values
            
        Returns:
            EngineConfig instance
        """
        valid_keys = {
            'enable_caching', 'max_cache_pages',
            'enable_text_processor', 'enable_image_processor', 
            'enable_content_modifier', 'enable_vector_processor',
            'timeout_seconds', 'max_file_size_mb',
            'validate_on_open', 'strict_mode',
            'log_level', 'enable_debug_logging'
        }
        
        # Filter to valid keys and warn about unknown keys
        filtered_config = {}
        for key, value in config.items():
            if key in valid_keys:
                filtered_config[key] = value
            else:
                logger.warning(f"Unknown config key '{key}' will be ignored")
        
        return cls(**filtered_config)
    
    @classmethod
    def default(cls) -> 'EngineConfig':
        """Create configuration with default values."""
        return cls()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EngineConfig("
            f"caching={self.enable_caching}, "
            f"text={self.enable_text_processor}, "
            f"image={self.enable_image_processor}, "
            f"modifier={self.enable_content_modifier}, "
            f"vector={self.enable_vector_processor}, "
            f"timeout={self.timeout_seconds}s)"
        )


@dataclass
class PageRange:
    """
    Represents a range of pages to process in a PDF document.
    
    Provides validation, normalization, and conversion to explicit page lists.
    Uses 1-based page numbering consistent with PDF specification.
    
    Example:
        >>> # Process first 10 pages
        >>> page_range = PageRange(start=1, end=10)
        >>> 
        >>> # Process from page 5 to end of document
        >>> page_range = PageRange(start=5, end=None)
        >>> 
        >>> # Process single page
        >>> page_range = PageRange.single_page(7)
    """
    
    start: int  # 1-based page number
    end: Optional[int] = None  # None means "to end of document"
    
    def __post_init__(self):
        """Validate page range on construction."""
        if self.start < 1:
            raise ValueError(f"start page must be >= 1, got {self.start}")
        
        if self.end is not None:
            if self.end < 1:
                raise ValueError(f"end page must be >= 1, got {self.end}")
            if self.end < self.start:
                raise ValueError(
                    f"end page ({self.end}) must be >= start page ({self.start})"
                )
    
    def to_page_numbers(self, total_pages: int) -> List[int]:
        """
        Convert range to explicit list of page numbers.
        
        Args:
            total_pages: Total number of pages in document
            
        Returns:
            List of 1-based page numbers to process
            
        Example:
            >>> page_range = PageRange(start=2, end=5)
            >>> page_range.to_page_numbers(10)
            [2, 3, 4, 5]
        """
        if total_pages < 1:
            return []
        
        # Clamp start to document bounds
        start = max(1, min(self.start, total_pages))
        
        # Resolve end page
        if self.end is None:
            end = total_pages
        else:
            end = min(self.end, total_pages)
        
        # Ensure start <= end after clamping
        if start > end:
            return []
        
        return list(range(start, end + 1))
    
    def validate(self, total_pages: int) -> bool:
        """
        Validate that range is within document bounds.
        
        Args:
            total_pages: Total number of pages in document
            
        Returns:
            True if range is valid for document
        """
        if total_pages < 1:
            logger.error("total_pages must be >= 1")
            return False
        
        if self.start > total_pages:
            logger.error(
                f"start page {self.start} exceeds total pages {total_pages}"
            )
            return False
        
        # If end is specified, validate it doesn't exceed document
        if self.end is not None and self.end > total_pages:
            logger.error(
                f"end page {self.end} exceeds total pages {total_pages}"
            )
            return False
        
        return True
    
    @classmethod
    def all_pages(cls) -> 'PageRange':
        """Create range representing all pages in document."""
        return cls(start=1, end=None)
    
    @classmethod
    def single_page(cls, page_num: int) -> 'PageRange':
        """
        Create range for a single page.
        
        Args:
            page_num: 1-based page number
        """
        return cls(start=page_num, end=page_num)
    
    @classmethod
    def from_tuple(cls, page_tuple: tuple) -> 'PageRange':
        """
        Create from tuple (start, end).
        
        Args:
            page_tuple: Tuple of (start, end) or (start,)
        """
        if len(page_tuple) == 1:
            return cls.single_page(page_tuple[0])
        elif len(page_tuple) == 2:
            return cls(start=page_tuple[0], end=page_tuple[1])
        else:
            raise ValueError(f"Invalid page tuple: {page_tuple}")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.end is None:
            return f"PageRange({self.start}→end)"
        elif self.start == self.end:
            return f"PageRange(page {self.start})"
        else:
            return f"PageRange({self.start}→{self.end})"
    
    def __len__(self) -> int:
        """
        Return number of pages in range (if end is specified).
        
        Raises ValueError if end is None (unbounded range).
        """
        if self.end is None:
            raise ValueError("Cannot get length of unbounded range")
        return self.end - self.start + 1
