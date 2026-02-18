"""
Content Modifier Processor

Handles PDF content removal operations including text and image deletion.
Follows the processor composition pattern established in Phase 1-3.
"""

import io
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
from pikepdf import Pdf, Page, parse_content_stream, unparse_content_stream

from engine.base_processor import BaseProcessor
from processors.pdf_graphics import GraphicsStateTracker, normalize_operator
from constants.pdf_operators import (
    TEXT_SHOWING_OPS,
    XOBJECT_OPS,
    PATH_CONSTRUCTION_OPS,
    PATH_PAINTING_OPS,
    SHADING_OPS,
    OP_MOVETO,
    OP_LINETO,
    OP_CURVETO,
    OP_CURVETO_V,
    OP_CURVETO_Y,
    OP_RECTANGLE
)
from utils.pdf_transforms import get_text_pivot_point, calculate_image_bbox

if TYPE_CHECKING:
    from engine.pdf_engine import PDFEngine

logger = logging.getLogger(__name__)

@dataclass
class ContentModifierOptions:
    """Configuration options for ContentModifier."""
    enable_reference_counting: bool = True
    overlap_threshold: float = 0.8
    text_position_tolerance: float = 1.0
    preserve_empty_content: bool = False
    
    def validate(self) -> bool:
        """Validate configuration options."""
        if not 0.0 <= self.overlap_threshold <= 1.0:
            logger.error("overlap_threshold must be between 0.0 and 1.0")
            return False
        
        if self.text_position_tolerance < 0:
            logger.error("text_position_tolerance must be non-negative")
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'enable_reference_counting': self.enable_reference_counting,
            'overlap_threshold': self.overlap_threshold,
            'text_position_tolerance': self.text_position_tolerance,
            'preserve_empty_content': self.preserve_empty_content
        }
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'ContentModifierOptions':
        """Create options from dictionary."""
        return cls(
            enable_reference_counting=config.get('enable_reference_counting', True),
            overlap_threshold=config.get('overlap_threshold', 0.8),
            text_position_tolerance=config.get('text_position_tolerance', 1.0),
            preserve_empty_content=config.get('preserve_empty_content', False)
        )


class ContentModifier(BaseProcessor):
    """
    Processor for PDF content removal operations.
    
    Handles:
    - Text removal by bounding box
    - Image removal by name and position
    - Resource cleanup and reference counting
    - Content stream parsing and reconstruction
    """
    
    def __init__(self, engine: 'PDFEngine', options: Optional[ContentModifierOptions] = None):
        """
        Initialize ContentModifier.
        
        Args:
            engine: Reference to parent PDFEngine
            options: Configuration options (uses defaults if None)
        """
        super().__init__(engine)
        self.options = options or ContentModifierOptions()
        
        # Validate options
        if not self.options.validate():
            raise ValueError("Invalid ContentModifierOptions")
    
    def initialize(self) -> None:
        """Initialize processor resources."""
        pass
    
    def cleanup(self) -> None:
        """Clean up processor resources."""
        pass
    
    def remove_content_from_page(
        self,
        page_index: int,
        text_bboxes: Optional[List] = None,
        image_bboxes: Optional[List] = None,
        vector_bboxes: Optional[List] = None
    ) -> None:
        """
        Remove content from a single page.
        
        Args:
            page_index: 0-based page index
            text_bboxes: List of text bounding boxes to remove
            image_bboxes: List of ImageRemovalRequest objects
            vector_bboxes: List of vector bounding boxes to remove
        """
        if not self.engine.is_open:
            raise RuntimeError("Engine must be open to modify content")
        
        # Get pikepdf page
        pikepdf_doc = self.engine.pikepdf_document
        if pikepdf_doc is None:
            raise RuntimeError("pikepdf document not available")
        
        page = pikepdf_doc.pages[page_index]
        page_num = page_index + 1  # For logging
        
        # Create page modifier
        modifier = PageContentModifier(
            pdf=pikepdf_doc,
            page=page,
            text_bboxes=text_bboxes or [],
            image_bboxes=image_bboxes or [],
            vector_bboxes=vector_bboxes or [],
            page_num=page_num,
            options=self.options
        )
        
        # Process the page
        modifier.process_and_replace_content()
    
    def remove_content_from_pages(
        self,
        page_removals: Dict[int, Dict]
    ) -> None:
        """
        Remove content from multiple pages.
        
        Args:
            page_removals: Dict mapping page_index to removal data
                          Example: {0: {'text_bboxes': [...], 'image_bboxes': [...], 'vector_bboxes': [...]}}
        """
        if not self.engine.is_open:
            raise RuntimeError("Engine must be open to modify content")
        
        logger.info(f"Processing content removal for {len(page_removals)} page(s)")
        
        for page_index, removal_data in page_removals.items():
            text_bboxes = removal_data.get('text_bboxes', [])
            image_bboxes = removal_data.get('image_bboxes', [])
            vector_bboxes = removal_data.get('vector_bboxes', [])
            
            self.remove_content_from_page(
                page_index=page_index,
                text_bboxes=text_bboxes,
                image_bboxes=image_bboxes,
                vector_bboxes=vector_bboxes
            )
    
    def get_modified_pdf_bytes(self) -> bytes:
        """
        Get the modified PDF as bytes.
        
        Returns:
            PDF file as bytes
        """
        if not self.engine.is_open:
            raise RuntimeError("Engine must be open to get PDF bytes")
        
        pikepdf_doc = self.engine.pikepdf_document
        if pikepdf_doc is None:
            raise RuntimeError("pikepdf document not available")
        
        # Save to buffer
        buffer = io.BytesIO()
        pikepdf_doc.save(buffer)
        buffer.seek(0)
        
        result_bytes = buffer.getvalue()
        # Only log size in debug
        logger.debug(f"Generated modified PDF: {len(result_bytes)} bytes")
        
        return result_bytes


class PageContentModifier:
    """
    Internal class for modifying a single PDF page.
    
    This encapsulates the low-level content stream parsing and filtering logic.
    Extracted from the legacy PDFPageModifier class.
    """
    
    def __init__(
        self,
        pdf: Pdf,
        page: Page,
        text_bboxes: List,
        image_bboxes: List,
        vector_bboxes: List,
        page_num: int,
        options: ContentModifierOptions
    ):
        """
        Initialize page modifier.
        
        Args:
            pdf: The PDF object
            page: The page to process
            text_bboxes: List of text bounding boxes to remove
            image_bboxes: List of ImageRemovalRequest objects
            vector_bboxes: List of vector bounding boxes to remove
            page_num: Page number for logging
            options: Modifier options
        """
        self.pdf = pdf
        self.page = page
        self.text_bboxes_to_remove = text_bboxes
        self.image_bboxes_to_remove = image_bboxes
        self.vector_bboxes_to_remove = vector_bboxes
        self.page_num = page_num
        self.options = options
        
        # Reference counting for safe image resource removal
        self.image_usage_counts = {}
        self.removed_image_counts = {}
        
        # Calculate page height for coordinate transformation
        self.page_height = 792.0  # Default letter size height
        try:
            if hasattr(page, 'MediaBox') and page.MediaBox:
                self.page_height = float(page.MediaBox[3] - page.MediaBox[1])
        except Exception as e:
            logger.warning(f"Could not get page height for page {page_num}, using default: {e}")
        
        # Initialize graphics state tracker
        self.graphics_state = GraphicsStateTracker(self.page_height)
    
    def process_and_replace_content(self) -> None:
        """
        Main method to process the page content and replace it with filtered content.
        """
        # Process content streams if they exist
        if '/Contents' in self.page:
            try:
                self._process_content_streams()
            except Exception as e:
                logger.error(f"Failed to process content streams on page {self.page_num}: {e}")
        
        # Remove image XObjects if specified
        if self.image_bboxes_to_remove and self.options.enable_reference_counting:
            self._remove_image_xobjects()
    
    def _process_content_streams(self) -> None:
        """Process all content streams on the page to remove targeted content."""
        try:
            # Parse content streams
            original_operators = parse_content_stream(self.page)
            
            # PASS 1: Identify vectors to remove and count image usage
            paths_to_remove = set()  # Store indices of path sequences to remove
            shadings_to_remove = set()  # Store indices of shading operators
            form_xobjects_to_remove = set()  # Store indices of Form XObject Do operators
            
            temp_graphics_state = GraphicsStateTracker(self.page_height)
            path_start_idx = None
            current_path_bbox = None
            
            for idx, operator in enumerate(original_operators):
                op_name_bytes = normalize_operator(operator)
                
                # Track XObject usage (images and forms)
                if op_name_bytes in XOBJECT_OPS and operator.operands:
                    xobject_name = str(operator.operands[0]).lstrip('/')
                    if self._verify_xobject_is_image(xobject_name):
                        self.image_usage_counts[xobject_name] = \
                            self.image_usage_counts.get(xobject_name, 0) + 1
                    elif self.vector_bboxes_to_remove and self._verify_xobject_is_form(xobject_name):
                        # Form XObject - check if it should be removed (opacity-affected vectors)
                        form_bbox = self._calculate_form_bbox_from_ctm(xobject_name, temp_graphics_state)
                        if form_bbox and self._bbox_matches_any_removal_request(form_bbox, self.vector_bboxes_to_remove):
                            form_xobjects_to_remove.add(idx)
                
                # Track shading operators for vector removal
                if self.vector_bboxes_to_remove and op_name_bytes in SHADING_OPS:
                    shading_bbox = self._calculate_shading_bbox_from_ctm(temp_graphics_state)
                    if shading_bbox and self._bbox_matches_any_removal_request(shading_bbox, self.vector_bboxes_to_remove):
                        shadings_to_remove.add(idx)
                
                # Track path construction for vector removal
                if self.vector_bboxes_to_remove:
                    if op_name_bytes in PATH_CONSTRUCTION_OPS:
                        # Mark start of path sequence
                        if path_start_idx is None:
                            path_start_idx = idx
                            current_path_bbox = None
                        # Update path bounding box
                        current_path_bbox = self._update_path_bbox_for_operator(
                            operator, temp_graphics_state, current_path_bbox
                        )
                    elif op_name_bytes in PATH_PAINTING_OPS:
                        # Decision point: check if path should be removed
                        if path_start_idx is not None and current_path_bbox:
                            if self._bbox_matches_any_removal_request(current_path_bbox, self.vector_bboxes_to_remove):
                                # Mark entire path sequence for removal (construction + painting)
                                paths_to_remove.update(range(path_start_idx, idx + 1))
                        # Reset path tracking
                        path_start_idx = None
                        current_path_bbox = None
                
                # Update graphics state for accurate CTM tracking
                temp_graphics_state._update_graphics_state(operator)
            
            # PASS 2: Filter operators based on removal decisions
            operators_to_keep = []
            removed_count = 0
            
            for idx, operator in enumerate(original_operators):
                # Skip operators marked for vector/shading/form removal
                if idx in paths_to_remove or idx in shadings_to_remove or idx in form_xobjects_to_remove:
                    continue
                
                should_keep = self._should_keep_operator(operator)
                
                if should_keep:
                    operators_to_keep.append(operator)
                    self.graphics_state._update_graphics_state(operator)
                else:
                    removed_count += 1
            
            if removed_count > 0:
                logger.debug(f"Page {self.page_num}: Removed {removed_count} operators during content modification")
            
            # Replace content stream
            if operators_to_keep:
                new_binary_data = unparse_content_stream(operators_to_keep)
                self.page.Contents = self.pdf.make_stream(new_binary_data)
            else:
                # Handle empty content stream
                if self.options.preserve_empty_content:
                    self.page.Contents = self.pdf.make_stream(b'')
                else:
                    # Remove Contents key entirely
                    if '/Contents' in self.page:
                        del self.page['/Contents']
                    logger.debug(f"Removed empty content stream from page {self.page_num}")
        
        except Exception as e:
            logger.error(f"Error processing content streams on page {self.page_num}: {e}")
            raise
    
    def _should_keep_operator(self, operator) -> bool:
        """Determine whether to keep a PDF operator based on removal criteria."""
        op_name_bytes = normalize_operator(operator)
        
        # Check for image removal
        if op_name_bytes in XOBJECT_OPS and operator.operands and self.image_bboxes_to_remove:
            xobject_name = str(operator.operands[0]).lstrip('/')
            
            for image_request in self.image_bboxes_to_remove:
                if image_request.name == xobject_name:
                    if self._verify_xobject_is_image(xobject_name):
                        # Get current transformation matrix
                        current_ctm = [
                            self.graphics_state.ctm[0,0], self.graphics_state.ctm[1,0],
                            self.graphics_state.ctm[0,1], self.graphics_state.ctm[1,1],
                            self.graphics_state.ctm[0,2], self.graphics_state.ctm[1,2]
                        ]
                        calculated_bbox = self._calculate_image_bbox(current_ctm)
                        
                        # Compare calculated position with requested bbox
                        if self._bboxes_overlap(calculated_bbox, image_request.bbox):
                            # Track this removal for reference counting
                            self.removed_image_counts[xobject_name] = \
                                self.removed_image_counts.get(xobject_name, 0) + 1
                            return False
        
        # Check for text removal
        if op_name_bytes in TEXT_SHOWING_OPS and self.text_bboxes_to_remove and \
           self.graphics_state.in_text_object:
            # Get text transform
            base_x, base_y, angle = self.graphics_state.get_text_transform()
            font_size = self.graphics_state.font_size
            
            for bbox in self.text_bboxes_to_remove:
                if self._point_in_bbox(base_x, base_y, font_size, angle, bbox):
                    return False
        
        return True
    
    def _verify_xobject_is_image(self, xobject_name: str) -> bool:
        """Verify that an XObject is actually an image."""
        try:
            if not hasattr(self.page, 'Resources') or '/XObject' not in self.page.Resources:
                return False
            
            xobjects = self.page.Resources.XObject
            
            # Try different name variations
            for name_variant in [xobject_name, f"/{xobject_name}"]:
                if name_variant in xobjects:
                    xobj = xobjects[name_variant]
                    return hasattr(xobj, 'Subtype') and xobj.Subtype == '/Image'
            
            return False
        except Exception as e:
            return False
    
    def _calculate_image_bbox(self, ctm) -> dict:
        """Calculate the actual bounding box of an image based on CTM."""
        try:
            return calculate_image_bbox(list(ctm), self.page_height)
        except Exception as e:
            logger.error(f"Error calculating image bbox: {e}")
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def _bboxes_overlap(self, calculated_bbox: dict, requested_bbox) -> bool:
        """Check if two bounding boxes overlap significantly."""
        try:
            # Get coordinates
            calc_x, calc_y = calculated_bbox['x'], calculated_bbox['y']
            calc_w, calc_h = calculated_bbox['width'], calculated_bbox['height']
            
            req_x, req_y = requested_bbox.x, requested_bbox.y
            req_w, req_h = requested_bbox.width, requested_bbox.height
            
            # Calculate intersection
            left = max(calc_x, req_x)
            top = max(calc_y, req_y)
            right = min(calc_x + calc_w, req_x + req_w)
            bottom = min(calc_y + calc_h, req_y + req_h)
            
            if left >= right or top >= bottom:
                return False
            
            intersection_area = (right - left) * (bottom - top)
            
            # Calculate union
            calc_area = calc_w * calc_h
            req_area = req_w * req_h
            union_area = calc_area + req_area - intersection_area
            
            if union_area == 0:
                return False
            
            # Calculate intersection over union (IoU)
            iou = intersection_area / union_area
            
            return iou >= self.options.overlap_threshold
        
        except Exception as e:
            logger.error(f"Error calculating bbox overlap: {e}")
            return False
    
    def _point_in_bbox(
        self,
        base_x: float,
        base_y: float,
        font_size: float,
        angle: float,
        bbox
    ) -> bool:
        """Check if a point is within a bounding box."""
        try:
            est_top_left_x, est_top_left_y = get_text_pivot_point(base_x, base_y, font_size, angle)
            
            # Convert to Y-down coordinate system
            point_x = est_top_left_x
            point_y = self.page_height - est_top_left_y
            
            # Check if point is in bbox
            tolerance = self.options.text_position_tolerance
            in_x_range = (bbox.x - tolerance) <= point_x <= (bbox.x + bbox.width + tolerance)
            in_y_range = (bbox.y - tolerance) <= point_y <= (bbox.y + bbox.height + tolerance)
            
            return in_x_range and in_y_range
        
        except AttributeError:
            # Graceful fallback for malformed bbox
            bbox_left = getattr(bbox, 'x', 0)
            bbox_top = getattr(bbox, 'y', 0)
            in_x_range = bbox_left <= base_x <= (bbox_left + getattr(bbox, 'width', 0))
            in_y_range = bbox_top <= (self.page_height - base_y) <= \
                         (bbox_top + getattr(bbox, 'height', 0))
            return in_x_range and in_y_range
    
    def _remove_image_xobjects(self) -> None:
        """Remove image XObjects from page resources if all instances removed."""
        if not self.removed_image_counts:
            return
        
        if not hasattr(self.page, 'Resources') or '/XObject' not in self.page.Resources:
            return
        
        xobjects = self.page.Resources.XObject
        final_removed_count = 0
        
        for img_name, removed_count in self.removed_image_counts.items():
            total_usage = self.image_usage_counts.get(img_name, 0)
            
            if removed_count >= total_usage:
                names_to_try = [img_name, f"/{img_name}"]
                for name_variant in names_to_try:
                    if name_variant in xobjects:
                        if hasattr(xobjects[name_variant], 'Subtype') and \
                           xobjects[name_variant].Subtype == '/Image':
                            del xobjects[name_variant]
                            final_removed_count += 1
                            break
        
        if final_removed_count > 0:
            logger.debug(f"Removed {final_removed_count} unused image resource(s) from page {self.page_num}")
    
    def _verify_xobject_is_form(self, xobject_name: str) -> bool:
        """Verify that an XObject is a Form XObject (often used for opacity-affected vectors)."""
        try:
            if not hasattr(self.page, 'Resources') or '/XObject' not in self.page.Resources:
                return False
            
            xobjects = self.page.Resources.XObject
            
            # Try different name variations
            for name_variant in [xobject_name, f"/{xobject_name}"]:
                if name_variant in xobjects:
                    xobj = xobjects[name_variant]
                    return hasattr(xobj, 'Subtype') and xobj.Subtype == '/Form'
            
            return False
        except Exception:
            return False
    
    def _bbox_matches_any_removal_request(self, calculated_bbox: dict, removal_bboxes: List) -> bool:
        """Check if calculated bbox matches any removal request using adaptive matching."""
        import math
        
        for bbox in removal_bboxes:
            try:
                # Get coordinates
                calc_x, calc_y = calculated_bbox['x'], calculated_bbox['y']
                calc_w, calc_h = calculated_bbox['width'], calculated_bbox['height']
                
                req_x, req_y = bbox.x, bbox.y
                req_w, req_h = bbox.width, bbox.height
                
                # Calculate areas
                calc_area = calc_w * calc_h
                req_area = req_w * req_h
                min_area = min(calc_area, req_area)
                max_area = max(calc_area, req_area)
                
                # TIER 1: Very thin elements (lines) - use center-point distance
                is_very_thin = (min(calc_w, calc_h) < 5 or min(req_w, req_h) < 5)
                if is_very_thin:
                    calc_center_x = calc_x + calc_w / 2
                    calc_center_y = calc_y + calc_h / 2
                    req_center_x = req_x + req_w / 2
                    req_center_y = req_y + req_h / 2
                    
                    distance = math.sqrt((calc_center_x - req_center_x)**2 + (calc_center_y - req_center_y)**2)
                    max_dimension = max(calc_w, calc_h, req_w, req_h)
                    threshold_distance = max_dimension * 0.75
                    
                    if distance <= threshold_distance:
                        return True
                
                # TIER 2: Small elements - center-point matching
                is_small_element = min_area < 300
                if is_small_element:
                    calc_center_x = calc_x + calc_w / 2
                    calc_center_y = calc_y + calc_h / 2
                    req_center_x = req_x + req_w / 2
                    req_center_y = req_y + req_h / 2
                    
                    distance = math.sqrt((calc_center_x - req_center_x)**2 + (calc_center_y - req_center_y)**2)
                    max_dimension = max(calc_w, calc_h, req_w, req_h)
                    threshold_distance = max_dimension * 0.6
                    
                    if distance <= threshold_distance:
                        return True
                
                # TIER 3: Standard IoU matching
                left = max(calc_x, req_x)
                top = max(calc_y, req_y)
                right = min(calc_x + calc_w, req_x + req_w)
                bottom = min(calc_y + calc_h, req_y + req_h)
                
                if left < right and top < bottom:
                    intersection_area = (right - left) * (bottom - top)
                    union_area = calc_area + req_area - intersection_area
                    
                    if union_area > 0:
                        iou = intersection_area / union_area
                        
                        # Adaptive threshold
                        if min_area < 500:
                            overlap_threshold = 0.4
                        elif min_area < 2000:
                            overlap_threshold = 0.55
                        else:
                            overlap_threshold = 0.7
                        
                        if iou >= overlap_threshold:
                            return True
                
                # TIER 4: Loose proximity fallback
                calc_center_x = calc_x + calc_w / 2
                calc_center_y = calc_y + calc_h / 2
                req_center_x = req_x + req_w / 2
                req_center_y = req_y + req_h / 2
                
                center_distance = math.sqrt((calc_center_x - req_center_x)**2 + (calc_center_y - req_center_y)**2)
                avg_size = math.sqrt((calc_w * calc_h + req_w * req_h) / 2)
                area_ratio = min_area / max_area if max_area > 0 else 0
                
                if center_distance < avg_size and area_ratio > 0.5:
                    return True
            
            except Exception:
                continue
        
        return False
    
    def _update_path_bbox_for_operator(self, operator, graphics_state, current_bbox: Optional[dict]) -> Optional[dict]:
        """Update the bounding box for a path based on a single operator."""
        try:
            op_name = normalize_operator(operator)
            operands = operator.operands
            
            # Extract coordinates based on operator type
            coords = []
            if op_name == OP_MOVETO or op_name == OP_LINETO:  # moveto, lineto
                if len(operands) >= 2:
                    coords = [(float(operands[0]), float(operands[1]))]
            elif op_name == OP_CURVETO:  # curveto
                if len(operands) >= 6:
                    coords = [
                        (float(operands[0]), float(operands[1])),
                        (float(operands[2]), float(operands[3])),
                        (float(operands[4]), float(operands[5]))
                    ]
            elif op_name == OP_CURVETO_V or op_name == OP_CURVETO_Y:  # curve variations
                if len(operands) >= 4:
                    coords = [
                        (float(operands[0]), float(operands[1])),
                        (float(operands[2]), float(operands[3]))
                    ]
            elif op_name == OP_RECTANGLE:  # rectangle
                if len(operands) >= 4:
                    x = float(operands[0])
                    y = float(operands[1])
                    width = float(operands[2])
                    height = float(operands[3])
                    coords = [(x, y), (x + width, y + height)]
            
            if not coords:
                return current_bbox
            
            # Transform coordinates using CTM
            transformed_coords = []
            for x, y in coords:
                tx = graphics_state.ctm[0,0] * x + graphics_state.ctm[0,1] * y + graphics_state.ctm[0,2]
                ty = graphics_state.ctm[1,0] * x + graphics_state.ctm[1,1] * y + graphics_state.ctm[1,2]
                # Convert to frontend Y-down coordinate system
                ty = self.page_height - ty
                transformed_coords.append((tx, ty))
            
            # Update or create bbox with adaptive padding
            all_x = [c[0] for c in transformed_coords]
            all_y = [c[1] for c in transformed_coords]
            
            base_width = max(all_x) - min(all_x) if all_x else 0
            base_height = max(all_y) - min(all_y) if all_y else 0
            element_size = max(base_width, base_height)
            
            # Adaptive padding
            if element_size < 10:
                padding = 3.0
            elif element_size < 50:
                padding = 2.5
            else:
                padding = 2.0
            
            if op_name == OP_CURVETO:  # Extra padding for curves
                padding += 1.0
            
            if current_bbox is None:
                return {
                    'x': min(all_x) - padding,
                    'y': min(all_y) - padding,
                    'width': (max(all_x) - min(all_x)) + (2 * padding),
                    'height': (max(all_y) - min(all_y)) + (2 * padding)
                }
            else:
                # Expand existing bbox
                min_x = min(current_bbox['x'], min(all_x) - padding)
                min_y = min(current_bbox['y'], min(all_y) - padding)
                max_x = max(current_bbox['x'] + current_bbox['width'], max(all_x) + padding)
                max_y = max(current_bbox['y'] + current_bbox['height'], max(all_y) + padding)
                
                return {
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                }
        except Exception:
            return current_bbox
    
    def _calculate_shading_bbox_from_ctm(self, graphics_state) -> Optional[dict]:
        """Calculate approximate bounding box for shading operators based on current CTM."""
        try:
            ctm = graphics_state.ctm
            
            # Try unit square transformation (most common)
            unit_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            transformed_unit = []
            for x, y in unit_corners:
                tx = ctm[0,0] * x + ctm[0,1] * y + ctm[0,2]
                ty = ctm[1,0] * x + ctm[1,1] * y + ctm[1,2]
                ty = self.page_height - ty
                transformed_unit.append((tx, ty))
            
            xs = [c[0] for c in transformed_unit]
            ys = [c[1] for c in transformed_unit]
            
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            
            # If unit square gives very small/zero area, try page-scale
            if width < 1 or height < 1:
                page_corners = [
                    (0, 0),
                    (self.page_height, 0),
                    (self.page_height, self.page_height),
                    (0, self.page_height)
                ]
                transformed_page = []
                for x, y in page_corners:
                    tx = ctm[0,0] * x + ctm[0,1] * y + ctm[0,2]
                    ty = ctm[1,0] * x + ctm[1,1] * y + ctm[1,2]
                    ty = self.page_height - ty
                    transformed_page.append((tx, ty))
                
                xs = [c[0] for c in transformed_page]
                ys = [c[1] for c in transformed_page]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
            
            # Adaptive padding
            padding = 8.0 if (width < 50 or height < 50) else 5.0
            
            return {
                'x': min(xs) - padding,
                'y': min(ys) - padding,
                'width': width + (2 * padding),
                'height': height + (2 * padding)
            }
        except Exception:
            return None
    
    def _calculate_form_bbox_from_ctm(self, xobject_name: str, graphics_state) -> Optional[dict]:
        """Calculate bounding box for Form XObject based on current CTM and Form's BBox."""
        try:
            if not hasattr(self.page, 'Resources') or '/XObject' not in self.page.Resources:
                return None
            
            xobjects = self.page.Resources.XObject
            
            # Find the Form XObject
            xobj = None
            for name_variant in [xobject_name, f"/{xobject_name}"]:
                if name_variant in xobjects:
                    xobj = xobjects[name_variant]
                    break
            
            if xobj is None:
                return None
            
            # Get Form's BBox
            bbox = getattr(xobj, 'BBox', None)
            if bbox is None:
                return self._calculate_shading_bbox_from_ctm(graphics_state)
            
            # Form BBox is [x1 y1 x2 y2]
            form_x1, form_y1, form_x2, form_y2 = [float(v) for v in bbox]
            
            # Get Form's Matrix if it exists
            form_matrix = getattr(xobj, 'Matrix', None)
            if form_matrix is not None:
                fm = [float(v) for v in form_matrix]
            else:
                fm = [1, 0, 0, 1, 0, 0]  # Identity
            
            # Transform corners by form matrix, then by CTM
            ctm = graphics_state.ctm
            corners = [
                (form_x1, form_y1),
                (form_x2, form_y1),
                (form_x2, form_y2),
                (form_x1, form_y2)
            ]
            
            transformed = []
            for x, y in corners:
                # Apply form matrix
                fx = fm[0] * x + fm[2] * y + fm[4]
                fy = fm[1] * x + fm[3] * y + fm[5]
                
                # Apply CTM
                tx = ctm[0,0] * fx + ctm[0,1] * fy + ctm[0,2]
                ty = ctm[1,0] * fx + ctm[1,1] * fy + ctm[1,2]
                
                # Convert to frontend Y-down
                ty = self.page_height - ty
                transformed.append((tx, ty))
            
            xs = [c[0] for c in transformed]
            ys = [c[1] for c in transformed]
            
            padding = 5.0
            
            return {
                'x': min(xs) - padding,
                'y': min(ys) - padding,
                'width': (max(xs) - min(xs)) + (2 * padding),
                'height': (max(ys) - min(ys)) + (2 * padding)
            }
        except Exception:
            return None