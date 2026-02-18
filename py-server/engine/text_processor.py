"""Text Processor for PDFEngine

Handles text extraction operations including stream-order extraction,
rotated text normalization, and text grouping with position tracking.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

from engine.base_processor import BaseProcessor
from engine.config import ProcessorOptions
from models.pdf_types import (
    PdfElement,
    PdfImageElement,
    PdfObstacles,
    PdfTextElement,
    PdfVectorElement,
    ProcessingTextRun,
)
from processors.text_processing_helpers import PdfRotationMarker
from processors.stream_order_device import StreamOrderDevice
from processors.text_grouping import (
    DEFAULT_TEXT_CONFIG,
    group_text_elements,
)
import processors.text_normalizer as text_normalizer

logger = logging.getLogger(__name__)


class TextProcessorOptions(ProcessorOptions):
    """Configuration options for text processing"""
    
    def __init__(
        self,
        enable_rotated_text: bool = True,
        enable_text_grouping: bool = True,
        text_grouping_config: Optional[Dict] = None,
        rotation_threshold_degrees: float = 1.0,
        rotation_angle_tolerance: float = 5.0
    ):
        """
        Initialize text processor options.
        
        Args:
            enable_rotated_text: Whether to detect and extract rotated text
            enable_text_grouping: Whether to group text into structured elements
            text_grouping_config: Configuration for text grouping behavior
            rotation_threshold_degrees: Minimum rotation to consider text as rotated
            rotation_angle_tolerance: Tolerance for matching rotation angles
        """
        self.enable_rotated_text = enable_rotated_text
        self.enable_text_grouping = enable_text_grouping
        self.text_grouping_config = text_grouping_config or {}
        self.rotation_threshold_degrees = rotation_threshold_degrees
        self.rotation_angle_tolerance = rotation_angle_tolerance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'enable_rotated_text': self.enable_rotated_text,
            'enable_text_grouping': self.enable_text_grouping,
            'text_grouping_config': self.text_grouping_config,
            'rotation_threshold_degrees': self.rotation_threshold_degrees,
            'rotation_angle_tolerance': self.rotation_angle_tolerance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TextProcessorOptions':
        """Create from dictionary representation"""
        return cls(
            enable_rotated_text=data.get('enable_rotated_text', True),
            enable_text_grouping=data.get('enable_text_grouping', True),
            text_grouping_config=data.get('text_grouping_config', {}),
            rotation_threshold_degrees=data.get('rotation_threshold_degrees', 1.0),
            rotation_angle_tolerance=data.get('rotation_angle_tolerance', 5.0),
        )


class TextProcessor(BaseProcessor):
    """
    Text extraction processor for PDFEngine.
    
    Handles all text extraction operations including horizontal text,
    rotated text, and text grouping/structuring.
    """
    
    def __init__(self, engine: 'PDFEngine', options: Optional[TextProcessorOptions] = None):
        """
        Initialize text processor.
        
        Args:
            engine: Parent PDFEngine instance
            options: TextProcessorOptions or None for defaults
        """
        super().__init__(engine)
        self.options = options or TextProcessorOptions()
        
        # State tracking
        self._page_obstacles: Dict[int, PdfObstacles] = {}
    
    def initialize(self) -> None:
        """Initialize text processor resources"""
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up text processor resources"""
        self._page_obstacles.clear()
        self._initialized = False
    
    def validate_state(self) -> bool:
        """Validate processor is in valid state"""
        if not self._initialized:
            logger.error("TextProcessor not initialized")
            return False
        return True
    
    def extract_stream_elements(
        self, 
        page_index: int, 
        position_strategy: Optional['PositionTrackingStrategy'] = None,
        skip_text_placeholders: bool = False,
        skip_image_placeholders: bool = False,
        skip_vector_placeholders: bool = False,
    ) -> List[Union[ProcessingTextRun, PdfImageElement, PdfVectorElement, PdfRotationMarker]]:
        """
        Extract page content in stream order with rotation markers.
        
        Args:
            page_index: 0-based page index
            position_strategy: Position tracking strategy (optional)
            skip_text_placeholders: Skip creating text elements (default: False)
            skip_image_placeholders: Skip creating image placeholders (default: False)
            skip_vector_placeholders: Skip creating vector placeholders (default: False)
            
        Returns:
            List of text runs, images, and rotation markers in stream order
        """
        page_num = page_index + 1
        page_height = self.engine.get_page_height(page_index)
        mediabox_bottom = self.engine.get_page_mediabox_bottom(page_index)
        
        # Extract obstacles for this page
        obstacles = self.engine.get_page_obstacles(page_index)
        self._page_obstacles[page_num] = obstacles
        
        # Create device with page height for coordinate transformations
        with open(self.engine.file_path, 'rb') as fp:
            rsrcmgr = PDFResourceManager()
            pdfminer_pages = list(PDFPage.get_pages(fp))
            
            device = StreamOrderDevice(
                rsrcmgr, 
                page_num, 
                page_height,
                mediabox_bottom=mediabox_bottom,
                position_strategy=position_strategy,
                skip_text_placeholders=skip_text_placeholders,
                skip_image_placeholders=skip_image_placeholders,
                skip_vector_placeholders=skip_vector_placeholders,
            )
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            device.interpreter = interpreter  # Allow device to recursively process Form XObjects
            pdfminer_page = pdfminer_pages[page_index]
            device.current_page = pdfminer_page  # Store page for Form XObject access
            interpreter.process_page(pdfminer_page)
            
            # Log final stream composition
            logger.debug(f"Page {page_num}: Stream processing complete - {len(device.elements)} elements")
            
            return device.elements
    
    def extract_rotated_text(self, page_index: int) -> List[ProcessingTextRun]:
        """
        Extract and normalize rotated text from a page.
        
        Args:
            page_index: 0-based page index
            
        Returns:
            List of normalized rotated text runs
        """
        if not self.options.enable_rotated_text:
            return []
        
        pike_page = self.engine.pikepdf_document.pages[page_index]
        page_num = page_index + 1
        page_height = self.engine.get_page_height(page_index)
        mediabox_bottom = self.engine.get_page_mediabox_bottom(page_index)
        
        rotated_runs = []
        try:
            # Normalize rotated text by physically rotating it in the PDF
            normalized_pdf_bytes = text_normalizer.normalize_rotated_text_on_page(pike_page, page_num)
            if normalized_pdf_bytes:
                logger.debug(f"Page {page_num}: Successfully normalized rotated text")
                
                # Use stream-order extraction on the normalized PDF
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(normalized_pdf_bytes)
                    tmp_path = tmp.name
                
                try:
                    # Import strategy here to avoid circular dependency
                    from processors.position_tracking import NormalizedRotatedStrategy
                    
                    # Extract from normalized PDF using stream-order device
                    with open(tmp_path, 'rb') as normalized_fp:
                        rsrcmgr = PDFResourceManager()
                        pdfminer_pages = list(PDFPage.get_pages(normalized_fp))
                        
                        if pdfminer_pages:
                            # Use NormalizedRotatedStrategy for sequential position advancement
                            device = StreamOrderDevice(
                                rsrcmgr, 
                                page_num, 
                                page_height,
                                mediabox_bottom=mediabox_bottom,
                                position_strategy=NormalizedRotatedStrategy()
                            )
                            device.accumulator.skip_rotation_check = True
                            interpreter = PDFPageInterpreter(rsrcmgr, device)
                            device.interpreter = interpreter
                            pdfminer_page = pdfminer_pages[0]
                            device.current_page = pdfminer_page
                            interpreter.process_page(pdfminer_page)
                            
                            # Extract text runs with look-ahead parsing for markers
                            i = 0
                            while i < len(device.elements):
                                elem = device.elements[i]
                                if isinstance(elem, ProcessingTextRun):
                                    # Check if next element is a rotation marker
                                    rotation = 0.0
                                    if i + 1 < len(device.elements):
                                        next_elem = device.elements[i + 1]
                                        if isinstance(next_elem, ProcessingTextRun) and '<|ORI:' in next_elem.text:
                                            # Parse marker from next element
                                            _, rotation = text_normalizer.parse_and_clean_run(next_elem.text)
                                            i += 1  # Skip the marker element
                                    
                                    # Clean text (in case marker is embedded)
                                    cleaned_text, embedded_rotation = text_normalizer.parse_and_clean_run(elem.text)
                                    # Use marker rotation if found, otherwise use embedded rotation
                                    final_rotation = rotation if rotation != 0.0 else embedded_rotation
                                    
                                    elem.text = cleaned_text
                                    elem.rotation = final_rotation
                                    
                                    if elem.text.strip():
                                        rotated_runs.append(elem)
                                i += 1
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Failed during rotated text extraction on page {page_num}: {e}")
        
        return rotated_runs
    
    def splice_rotated_text(
        self, 
        stream_elements: List[Union[ProcessingTextRun, PdfImageElement, PdfRotationMarker]],
        rotated_runs: List[ProcessingTextRun],
        page_num: int
    ) -> List[Union[ProcessingTextRun, PdfImageElement]]:
        """
        Replace rotation markers with normalized rotated text runs.
        
        Args:
            stream_elements: Stream elements with rotation markers
            rotated_runs: Normalized rotated text runs
            page_num: Page number (1-based)
            
        Returns:
            Stream elements with markers replaced by actual rotated text
        """
        if not any(isinstance(e, PdfRotationMarker) for e in stream_elements):
            return stream_elements
        
        # Group rotated runs by normalized angle
        rotated_by_angle = defaultdict(list)
        for run in rotated_runs:
            normalized_angle = self._normalize_angle(run.rotation)
            rotated_by_angle[normalized_angle].append(run)
        
        final_elements = []
        marker_index_by_angle = defaultdict(int)
        
        for element in stream_elements:
            if isinstance(element, PdfRotationMarker):
                marker_angle = self._normalize_angle(element.rotation)
                
                # Find matching angle group
                matched_angle = None
                for normalized_angle in rotated_by_angle.keys():
                    if self._angles_match(marker_angle, normalized_angle):
                        matched_angle = normalized_angle
                        break
                
                if matched_angle is not None:
                    runs_for_angle = rotated_by_angle[matched_angle]
                    current_index = marker_index_by_angle[matched_angle]
                    
                    remaining_runs = runs_for_angle[current_index:]
                    if remaining_runs:
                        final_elements.extend(remaining_runs)
                        marker_index_by_angle[matched_angle] = len(runs_for_angle)
            else:
                final_elements.append(element)
        
        return final_elements
    
    def group_content(
        self, 
        stream_elements: List[Union[ProcessingTextRun, PdfImageElement, PdfVectorElement]], 
        config: Optional[Dict] = None,
        page_num: Optional[int] = None
    ) -> List[PdfElement]:
        """
        Group stream elements into structured content.
        
        Args:
            stream_elements: Text runs and images in stream order
            config: Configuration for text grouping behavior
            page_num: Page number for obstacle lookup (optional)
            
        Returns:
            List of grouped PdfTextElements and PdfImageElements
        """
        if not stream_elements:
            return []
        
        if not self.options.enable_text_grouping:
            return stream_elements
        
        # Merge user config with processor config
        final_config = {
            **DEFAULT_TEXT_CONFIG, 
            **self.options.text_grouping_config,
            **(config or {})
        }
        
        # Get obstacles for this page
        obstacles = self._page_obstacles.get(page_num) if page_num else None
        
        # Set stream_index on text runs to track original position
        for idx, el in enumerate(stream_elements):
            if isinstance(el, ProcessingTextRun):
                el.stream_index = idx
        
        # Separate images and vectors from text for grouping
        # Images & vectors: store index externally as (index, element) tuple
        text_runs = []
        images_with_index = []
        vectors_with_index = []
        for idx, el in enumerate(stream_elements):
            if isinstance(el, ProcessingTextRun):
                text_runs.append(el)
            elif isinstance(el, PdfImageElement):
                images_with_index.append((idx, el))
            elif isinstance(el, PdfVectorElement):
                vectors_with_index.append((idx, el))
        
        # Partition text by rotation (matching legacy behavior)
        partitions = defaultdict(list)
        for run in text_runs:
            if run.text.strip():
                angle = run.rotation
                partitions[angle].append(run)
        
        # Group text within each rotation partition
        grouped_text = []
        for angle, runs_in_partition in partitions.items():
            grouped_partition = group_text_elements(
                runs_in_partition,
                final_config,
                obstacles=obstacles
            )
            grouped_text.extend(grouped_partition)
        
        # Extract minimum stream_index from each grouped element to restore stream order
        grouped_with_index = []
        for gi, grouped_elem in enumerate(grouped_text):
            min_idx = float('inf')
            
            if isinstance(grouped_elem, PdfTextElement) and hasattr(grouped_elem, 'sections'):
                # Extract stream indices from all runs in all sections
                for section in grouped_elem.sections:
                    if hasattr(section, 'lines'):
                        for line in section.lines:
                            for run in line:
                                if hasattr(run, 'stream_index') and run.stream_index is not None:
                                    min_idx = min(min_idx, run.stream_index)
            
            # Fallback if no valid index found
            if min_idx == float('inf'):
                min_idx = 0
            
            grouped_with_index.append((min_idx, grouped_elem))
        
        # Combine and sort by stream index to restore stream order
        all_elements_with_index = grouped_with_index + images_with_index + vectors_with_index
        all_elements_with_index.sort(key=lambda x: x[0])
        
        final_elements = [elem for _, elem in all_elements_with_index]
        
        logger.debug(f"Grouped into {len(final_elements)} elements")
        
        return final_elements
    
    def get_page_stream(
        self, 
        page_index: int,
        position_strategy: Optional['PositionTrackingStrategy'] = None,
        skip_text_placeholders: bool = False,
        skip_image_placeholders: bool = False,
        skip_vector_placeholders: bool = False
    ) -> List[Union[ProcessingTextRun, PdfImageElement, PdfVectorElement]]:
        """
        Get complete page stream with rotated text integrated and images hydrated.
        
        Args:
            page_index: 0-based page index
            position_strategy: Position tracking strategy (optional)
            skip_text_placeholders: Skip creating text elements (default: False)
            skip_image_placeholders: Skip creating image placeholders (default: False)
            skip_vector_placeholders: Skip creating vector placeholders (default: False)
            
        Returns:
            Complete page stream with all content
        """
        page_num = page_index + 1
        
        # Get raw stream elements
        stream_elements = self.extract_stream_elements(
            page_index, 
            position_strategy,
            skip_text_placeholders=skip_text_placeholders,
            skip_image_placeholders=skip_image_placeholders,
            skip_vector_placeholders=skip_vector_placeholders

        )
        
        # Check for rotation markers
        has_markers = any(isinstance(e, PdfRotationMarker) for e in stream_elements)
        
        # Handle rotated text if present
        if has_markers and self.options.enable_rotated_text:
            logger.debug(f"Page {page_num}: Rotation markers detected, extracting rotated text")
            rotated_runs = self.extract_rotated_text(page_index)
            stream_elements = self.splice_rotated_text(stream_elements, rotated_runs, page_num)
        
        # Hydrate images using ImageProcessor if available
        try:
            image_processor = self.engine.image_processor
            stream_elements = image_processor.hydrate_images_in_stream(stream_elements, page_index)
        except RuntimeError:
            pass
        
        return stream_elements
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180) range"""
        while angle >= 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return round(angle, 1)
    
    def _angles_match(self, angle1: float, angle2: float, tolerance: Optional[float] = None) -> bool:
        """Check if two angles match within tolerance"""
        if tolerance is None:
            tolerance = self.options.rotation_angle_tolerance
        
        angle1 = self._normalize_angle(angle1)
        angle2 = self._normalize_angle(angle2)
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff <= tolerance