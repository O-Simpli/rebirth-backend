"""Stream Order Device for PDF Content Extraction

PDFMiner device that processes PDF content streams in paint order,
extracting text and images as they appear in the PDF stream.

This device builds character data from PDFMiner callbacks with full
graphics state tracking for accurate positioning and transformation.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix

from models.pdf_types import (
    PdfImageElement,
    PdfVectorElement,
    ProcessingTextRun,
)
from processors.text_processing_helpers import (
    PdfRotationMarker,
    StyledRun,
    TextRunAccumulator,
    create_processing_text_run,
    _get_char_orientation,
    _is_rotated_char,
)
from utils.pdf_transforms import calculate_image_bbox, decompose_ctm
from processors.position_tracking import (
    PositionTrackingStrategy,
    RegularTextStrategy,
)

logger = logging.getLogger(__name__)

# Constants (imported from pdf_engine)
COORDINATE_PRECISION = 2
BASELINE_TOLERANCE = 3.0
ROTATION_THRESHOLD_DEGREES = 1.0
ROTATION_ANGLE_TOLERANCE = 5.0
POSITION_BACKWARDS_TOLERANCE = 1.0
SAME_POSITION_TOLERANCE = 0.1
FONT_SIZE_MULTIPLIER_FORWARD_JUMP = 2.0
FONT_SIZE_TOLERANCE = 0.1
FAUX_BOLD_POSITION_TOLERANCE = 0.5
SCALING_PERCENTAGE_DIVISOR = 0.01
DISPLACEMENT_MULTIPLIER = 0.001
DEFAULT_FONT_ASCENT = 0.75
DEFAULT_FONT_DESCENT = -0.25
MARKER_FONT_SIZE = 0.01
MARKER_FONT_TOLERANCE = 0.001
TYPE3_FONT_UNITS = 1000.0
CAPHEIGHT_TO_ASCENT_RATIO = 1.2
CAPHEIGHT_TO_DESCENT_RATIO = -0.3
INVALID_METRIC_THRESHOLD = 0.001


class StreamOrderDevice(PDFDevice):
    """
    Stream extraction device processing PDF content in paint order.
    
    Extracts text and images as they appear in the content stream,
    building character data from PDFMiner callbacks with graphics state tracking.
    """
    
    def __init__(
        self, 
        rsrcmgr: PDFResourceManager, 
        page_num: int, 
        page_height: float,
        mediabox_bottom: float = 0.0,
        position_strategy: Optional[PositionTrackingStrategy] = None,
        skip_text_placeholders: bool = False,
        skip_image_placeholders: bool = False,
        skip_vector_placeholders: bool = False
    ):
        """
        Initialize stream order device for paint-order content extraction.
        
        Args:
            rsrcmgr: PDF resource manager
            page_num: Page number (1-indexed)
            page_height: Page height in points
            mediabox_bottom: MediaBox bottom Y-coordinate (default: 0.0)
                pdfminer coordinates are in raw PDF space, must subtract this offset
                to align with pikepdf coordinates that are MediaBox-relative
            position_strategy: Position tracking strategy for text
            skip_text_placeholders: Skip creating text elements
            skip_image_placeholders: Skip creating image placeholders
            skip_vector_placeholders: Skip creating vector placeholders
        """
        super().__init__(rsrcmgr)
        self.rsrcmgr = rsrcmgr
        self.page_num = page_num
        self.page_height = page_height
        self.mediabox_bottom = mediabox_bottom
        self.position_strategy = position_strategy or RegularTextStrategy()
        
        # Skip flags for selective extraction
        self.skip_text_placeholders = skip_text_placeholders
        self.skip_image_placeholders = skip_image_placeholders
        self.skip_vector_placeholders = skip_vector_placeholders
        
        # Output elements
        self.elements: List[Union[ProcessingTextRun, PdfImageElement, PdfVectorElement, PdfRotationMarker]] = []
        self.image_counter = 0
        self.vector_counter = 0
        self.marker_counter = 0
        self.render_string_count = 0
        
        # Graphics state tracking
        self.current_ctm = [1, 0, 0, 1, 0, 0]
        self.ctm_stack = []
        
        # DEBUG: Form XObject tracking
        self._form_xobject_depth = 0
        self._form_xobject_stack = []
        self._vectors_in_form_xobjects = 0
        
        # Text extraction state
        self.accumulator = TextRunAccumulator()
        self.current_rotated_segment: List[Dict] = []
        self.current_rotation_angle: Optional[float] = None
        self.last_text_position: Optional[Tuple[float, float]] = None
        self.interpreter = None
        self.current_page = None
        
        # Track text matrix advancement
        self.last_matrix_position: Optional[List[float]] = None
        self.render_string_call_index = 0
        
        # Faux-bold duplicate detection
        self._last_render_text: Optional[str] = None
        self._last_render_position: Optional[Tuple[float, float]] = None
        
        # Quote fix state
        self.pending_open_quote_run: Optional[ProcessingTextRun] = None
    
    def begin_page(self, page, ctm):
        """Initialize state for new page"""
        logger.debug(f"Page {self.page_num}: begin_page called")
        self.elements = []
        self.current_ctm = list(ctm) if ctm else [1, 0, 0, 1, 0, 0]
        self.ctm_stack = []
        self.current_rotated_segment = []
        self.current_rotation_angle = None
        self.current_page = page
    
    def end_page(self, page):
        """Flush any remaining text at page end"""
        if self.current_rotated_segment:
            self._flush_rotated_segment()
        self._flush_text()
    
    def set_ctm(self, ctm):
        """Update current transformation matrix"""
        self.current_ctm = list(ctm)
    
    def push_graphics_state(self):
        """Save graphics state (q operator)"""
        self.ctm_stack.append(list(self.current_ctm))
    
    def pop_graphics_state(self):
        """Restore graphics state (Q operator)"""
        if self.ctm_stack:
            self.current_ctm = self.ctm_stack.pop()
    
    def _add_to_rotated_segment(self, char_data: Dict):
        """Add character to rotated text segment"""
        rotation = _get_char_orientation(char_data)
        if self.current_rotated_segment and self.current_rotation_angle is not None:
            angle_diff = abs(rotation - self.current_rotation_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > ROTATION_ANGLE_TOLERANCE:
                self._flush_rotated_segment()
        if not self.current_rotated_segment:
            self.current_rotation_angle = rotation
        self.current_rotated_segment.append(char_data)
    
    def _flush_rotated_segment(self):
        """Flush accumulated rotated text as a marker"""
        if not self.current_rotated_segment:
            return
        self._flush_text()
        first_char = self.current_rotated_segment[0]
        angles = [_get_char_orientation(char) for char in self.current_rotated_segment]
        avg_angle = sum(angles) / len(angles)
        marker = PdfRotationMarker(
            page_num=self.page_num,
            marker_id=self.marker_counter,
            x=first_char.get('x0', 0.0),
            y=first_char.get('top', 0.0),
            rotation=-avg_angle
        )
        self.marker_counter += 1
        self.elements.append(marker)
        self.current_rotated_segment = []
        self.current_rotation_angle = None
    
    def _flush_text(self):
        """Flush accumulated text run"""
        import re
        
        completed_runs = self.accumulator.flush()  # Returns List[StyledRun]
        if not completed_runs:
            return
        
        # Process each completed run
        for completed_run in completed_runs:
            
            processing_run = create_processing_text_run(completed_run)
            
            # Check for cross-run malformed quote patterns
            if self.pending_open_quote_run is not None:
                match = re.match(r'^([A-Za-z\s]{0,20}?)0(?=[\s,.)!?;:\n]|$)', processing_run.text)
                if match:
                    # Found closing pattern
                    prev_text = self.pending_open_quote_run.text
                    open_match = re.search(r'/([A-Za-z]+)$', prev_text)
                    if open_match:
                        self.pending_open_quote_run.text = re.sub(r'/([A-Za-z]+)$', r'"\1', prev_text)
                        processing_run.text = re.sub(r'^([A-Za-z\s]{0,20}?)0', r'\1"', processing_run.text)
                
                self.pending_open_quote_run = None
            
            # Check if this run ends with an unclosed opening quote
            if re.search(r'/[A-Za-z]+$', processing_run.text):
                self.pending_open_quote_run = processing_run
            
            # Only append if text extraction is enabled
            if not self.skip_text_placeholders:
                self.elements.append(processing_run)
    
    def render_string(self, textstate, seq, ncs, graphicstate):
        """Handle text rendering (Tj/TJ operators)"""
        self.render_string_count += 1
        
        # Capture text matrix position at start of render_string
        incoming_tm_x = textstate.matrix[4]
        
        # Store tm_start_x in accumulator for this render_string call
        if self.accumulator.current_run_data:
            if self.accumulator.current_run_data.get("tm_start_x") is None:
                self.accumulator.current_run_data["tm_start_x"] = incoming_tm_x
        
        font = textstate.font
        if not font:
            return
        
        # Only reconstruct text for checking faux-bold detection (avoid extensive logging)
        seq_text = ""
        for item in seq:
            if isinstance(item, bytes):
                try:
                    seq_text += item.decode('latin-1', errors='ignore')
                except:
                    pass
            elif isinstance(item, str):
                seq_text += item
        
        # Detect faux-bold duplicates
        current_position = (textstate.matrix[4], textstate.matrix[5])
        if (self._last_render_text is not None and self._last_render_position is not None and
            self._last_render_text == seq_text and 
            abs(self._last_render_position[0] - current_position[0]) < FAUX_BOLD_POSITION_TOLERANCE and
            abs(self._last_render_position[1] - current_position[1]) < FAUX_BOLD_POSITION_TOLERANCE):
            return
        
        self._last_render_text = seq_text
        self._last_render_position = current_position
        
        # Extract color from graphics state
        color_value = getattr(graphicstate, 'ncolor', None) if graphicstate else None
        
        fontsize = textstate.fontsize
        scaling = textstate.scaling * SCALING_PERCENTAGE_DIVISOR
        rise = textstate.rise
        fontname = font.fontname if hasattr(font, 'fontname') else "Unknown"
        
        # Track position manually for font subsets
        incoming_matrix = list(textstate.matrix)
        self.render_string_call_index += 1
        
        if self.last_matrix_position is not None:
            same_line = abs(incoming_matrix[5] - self.last_matrix_position[5]) < SAME_POSITION_TOLERANCE
            
            is_marker_font = abs(fontsize - MARKER_FONT_SIZE) < MARKER_FONT_TOLERANCE
            just_processed_marker = hasattr(self, '_last_was_marker') and self._last_was_marker
            
            accum_right_edge = self.accumulator.current_run_data['x1'] if self.accumulator.current_run_data else 0
            accum_font_size = self.accumulator.current_run_data['font_size'] if self.accumulator.current_run_data else fontsize
            close_to_accumulator = abs(incoming_matrix[4] - accum_right_edge) < max(fontsize, accum_font_size) * 2.0
            
            same_line_same_start = same_line and abs(incoming_matrix[4] - getattr(self, '_line_start_x', float('inf'))) < FONT_SIZE_MULTIPLIER_FORWARD_JUMP
            
            is_backwards = (incoming_matrix[4] < self.last_matrix_position[4] - POSITION_BACKWARDS_TOLERANCE and 
                           not same_line_same_start and not just_processed_marker and not close_to_accumulator)
            
            if not same_line:
                self._line_start_x = incoming_matrix[4]
            
            if is_backwards and self.accumulator.current_run_data:
                self._flush_text()
            
            # Delegate position tracking to strategy
            use_tracked = self.position_strategy.should_use_tracked_position(
                incoming_x=incoming_matrix[4],
                last_x=self.last_matrix_position[4],
                font_size=fontsize,
                same_line=same_line,
                is_backwards=is_backwards
            )
            matrix = list(self.last_matrix_position) if use_tracked else incoming_matrix
            
            self._last_was_marker = is_marker_font
        else:
            matrix = incoming_matrix
        
        # Flush if text position jumped significantly (skip if threshold is 0)
        if matrix and len(matrix) >= 6:
            current_text_pos = (matrix[4], matrix[5])
            if self.last_text_position:
                pos_delta = math.sqrt(
                    (current_text_pos[0] - self.last_text_position[0])**2 +
                    (current_text_pos[1] - self.last_text_position[1])**2
                )
                if fontsize > 0 and pos_delta > fontsize:
                    self._flush_text()
        
        # Process each item in the sequence
        for item in seq:
            if isinstance(item, (int, float)):
                # TJ array displacement
                displacement = -item * fontsize * scaling * DISPLACEMENT_MULTIPLIER
                matrix = mult_matrix([1, 0, 0, 1, displacement, 0], matrix)
                continue
            
            # Process bytes/string as characters
            if isinstance(item, bytes):
                chars_bytes = item
            elif isinstance(item, str):
                chars_bytes = item.encode('latin-1', errors='ignore')
            else:
                continue
            
            # Decode bytes to CIDs using font's CMap
            try:
                cids = font.decode(chars_bytes)
            except Exception as e:
                continue
            
            # Process each CID
            for cid in cids:
                # Decode CID to Unicode using font's ToUnicode CMap
                try:
                    text = font.to_unichr(cid)
                    if text is None or not text:
                        continue
                except Exception as e:
                    continue
                
                # Calculate character metrics
                try:
                    # Width calculation
                    char_width = font.char_width(cid) if hasattr(font, 'char_width') else 0
                    char_spacing = getattr(textstate, 'charspace', 0)
                    word_spacing = getattr(textstate, 'wordspace', 0)
                    
                    # Word spacing is only applied to space characters (ASCII 32)
                    adv = (char_width * fontsize * scaling) + char_spacing
                    if cid == 32 and word_spacing != 0:
                        adv += word_spacing
                    
                    # Height calculation
                    ascent = font.get_ascent() if hasattr(font, 'get_ascent') else DEFAULT_FONT_ASCENT
                    descent = font.get_descent() if hasattr(font, 'get_descent') else DEFAULT_FONT_DESCENT
                    
                    # Fallback for invalid metrics
                    ascent_invalid = abs(ascent) < INVALID_METRIC_THRESHOLD
                    descent_invalid = abs(descent) < INVALID_METRIC_THRESHOLD
                    
                    if ascent_invalid or descent_invalid:
                        fallback_applied = False
                        
                        # Try CapHeight from font descriptor
                        if hasattr(font, 'descriptor') and font.descriptor:
                            capheight = font.descriptor.get('CapHeight')
                            if capheight is not None and capheight > 0:
                                normalized_capheight = capheight / TYPE3_FONT_UNITS
                                
                                if ascent_invalid:
                                    ascent = normalized_capheight * CAPHEIGHT_TO_ASCENT_RATIO
                                if descent_invalid:
                                    descent = normalized_capheight * CAPHEIGHT_TO_DESCENT_RATIO
                                
                                fallback_applied = True
                        
                        # Ultimate fallback
                        if not fallback_applied:
                            if ascent_invalid:
                                ascent = DEFAULT_FONT_ASCENT
                            if descent_invalid:
                                descent = DEFAULT_FONT_DESCENT
                    
                    # Position calculation
                    tm_x0, tm_y0 = apply_matrix_pt(matrix, (0, rise))
                    tm_x1 = tm_x0 + adv
                    
                    # CTM transformation
                    x0, y0 = apply_matrix_pt(self.current_ctm, (tm_x0, tm_y0))
                    x1, _ = apply_matrix_pt(self.current_ctm, (tm_x1, tm_y0))
                    
                    # Calculate effective fontsize
                    text_matrix_y_scale = abs(matrix[3])
                    ctm_y_scale = abs(self.current_ctm[3])
                    effective_fontsize = fontsize * text_matrix_y_scale * ctm_y_scale
                    
                    # Calculate top and bottom
                    top_pdf = y0 + descent * effective_fontsize
                    bottom_pdf = y0 + ascent * effective_fontsize
                    
                    # Convert to Y-down coordinate system
                    top_frontend = self.page_height - bottom_pdf
                    bottom_frontend = self.page_height - top_pdf
                    
                    # Build character dict
                    combined_matrix = mult_matrix(self.current_ctm, matrix)
                    
                    char_data = {
                        "text": text,
                        "fontname": str(fontname),
                        "size": effective_fontsize,
                        "non_stroking_color": color_value,
                        "ncs": color_value,
                        "x0": x0,
                        "x1": x1,
                        "top": top_frontend,
                        "bottom": bottom_frontend,
                        "matrix": combined_matrix
                    }
                    
                    # Check for rotation
                    if _is_rotated_char(char_data):
                        self._add_to_rotated_segment(char_data)
                    else:
                        if self.current_rotated_segment:
                            self._flush_rotated_segment()
                        
                        # Add character to accumulator
                        completed_run = self.accumulator.add(char_data)
                        if completed_run:
                            processing_run = create_processing_text_run(completed_run)
                            # Only append if text extraction is enabled
                            if not self.skip_text_placeholders:
                                self.elements.append(processing_run)
                    
                    # Update matrix for next character
                    matrix = mult_matrix([1, 0, 0, 1, adv, 0], matrix)
                except Exception:
                    continue
        
        # Save final matrix position and store in accumulator
        if matrix and len(matrix) >= 6:
            self.last_text_position = (matrix[4], matrix[5])
            self.last_matrix_position = list(matrix)
            # Store tm_end_x in current run
            if self.accumulator.current_run_data:
                if self.accumulator.current_run_data.get("tm_start_x") is None:
                    self.accumulator.current_run_data["tm_start_x"] = incoming_tm_x
                self.accumulator.current_run_data["tm_end_x"] = matrix[4]

    
    def render_image(self, name, stream):
        """Process image rendering and create placeholder"""
        # Flush any accumulated text before image
        if self.current_rotated_segment:
            self._flush_rotated_segment()
        self._flush_text()
        
        # Decode image name
        image_name = name
        if isinstance(name, bytes):
            image_name = name.decode('latin-1', errors='ignore')
        if image_name.startswith('/'):
            image_name = image_name[1:]
        
        # Calculate image bounding box
        try:
            bbox = calculate_image_bbox(self.current_ctm, self.page_height, self.mediabox_bottom)
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        except Exception as e:
            logger.warning(f"Failed to calculate image bbox: {e}")
            x, y, width, height = 0, 0, 0, 0
        
        # Create placeholder
        self.image_counter += 1
        placeholder = PdfImageElement(
            id=f"img_placeholder_{self.image_counter}_{image_name}",
            type="image",
            name=image_name,
            imageIndex=self.image_counter - 1,
            x=round(x, COORDINATE_PRECISION),
            y=round(y, COORDINATE_PRECISION),
            width=round(width, COORDINATE_PRECISION),
            height=round(height, COORDINATE_PRECISION),
            mimeType="image/unknown",
            data=None
        )
        
        # Only append if image extraction is enabled
        if not self.skip_image_placeholders:
            self.elements.append(placeholder)
    
    def begin_figure(self, name, bbox, matrix):
        """Handle Form XObject - save CTM and process content"""
        self._form_xobject_depth += 1
        self._form_xobject_stack.append(name)
        
        self.push_graphics_state()
        if matrix:
            self.current_ctm = mult_matrix(self.current_ctm, matrix)
        
        # Process Form XObject's content stream
        if self.interpreter and self.current_page:
            try:
                if hasattr(self.current_page, 'resources') and self.current_page.resources:
                    resources = self.current_page.resources
                    if 'XObject' in resources:
                        xobjects = resources['XObject']
                        xobj_key = name if name in xobjects else f'/{name}' if f'/{name}' in xobjects else name.lstrip('/')
                        
                        if xobj_key in xobjects:
                            xobj = xobjects[xobj_key]
                            
                            from pdfminer.pdfinterp import PDFContentParser
                            from pdfminer.psparser import PSKeyword
                            from pdfminer.pdftypes import resolve1
                            
                            try:
                                xobj_resolved = resolve1(xobj)
                                
                                # Check if it's a Form XObject
                                if isinstance(xobj_resolved, dict) and xobj_resolved.get('Subtype') and str(xobj_resolved.get('Subtype')) == '/Form':
                                    
                                    # Get content stream
                                    if hasattr(xobj_resolved, 'get_data'):
                                        data = xobj_resolved.get_data()
                                    else:
                                        from pdfminer.pdftypes import PDFStream
                                        if isinstance(xobj_resolved, PDFStream):
                                            data = xobj_resolved.get_data()
                                        else:
                                            data = None
                                    
                                    if data:
                                        # Parse and execute content stream
                                        parser = PDFContentParser([data])
                                        operator_count = 0
                                        paint_operators = {'S', 's', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*'}
                                        
                                        for obj in parser:
                                            if isinstance(obj, PSKeyword):
                                                operator_count += 1
                                                self.interpreter.do_keyword(obj, parser.pop_args())
                            except Exception as e2:
                                logger.error(f"   ❌ Error processing Form XObject '{name}': {e2}", exc_info=True)
            except Exception as e:
                logger.error(f"   ❌ Error accessing Form XObject '{name}': {e}", exc_info=True)
    
    def end_figure(self, name):
        """End Form XObject - restore CTM"""
        self._form_xobject_depth -= 1
        if self._form_xobject_stack:
            self._form_xobject_stack.pop()
        
        self.pop_graphics_state()
    
    # Stub methods for PDFDevice protocol
    def begin_tag(self, tag, props=None): pass
    def end_tag(self): pass
    def do_tag(self, tag, props=None): pass
    
    def do_sh(self, name, raw_stream=None):
        """Handle shading operator (gradients) - create vector placeholder"""
        
        if self.skip_vector_placeholders:
            return
        
        # Flush any accumulated text before vector
        if self.current_rotated_segment:
            self._flush_rotated_segment()
        self._flush_text()
        
        # For shadings, we don't have path data, so we use CTM to estimate bbox
        x = self.current_ctm[4]
        y = self.current_ctm[5]
        
        # Flip Y coordinate to top-left origin
        y_flipped = self.page_height - (y - self.mediabox_bottom)
        
        # Create a generic placeholder - will be matched by position/size
        placeholder_width = 100.0
        placeholder_height = 100.0
        
        # Create vector placeholder
        self.vector_counter += 1
        placeholder = PdfVectorElement(
            id=f"vec_shading_{self.page_num}_{self.vector_counter}",
            type="vector",
            x=round(x, COORDINATE_PRECISION),
            y=round(y_flipped, COORDINATE_PRECISION),
            width=round(placeholder_width, COORDINATE_PRECISION),
            height=round(placeholder_height, COORDINATE_PRECISION),
            svgContent="",  # Will be hydrated later
            pathData="",    # Will be hydrated later
            opacity=1.0,
            stroke=None,
            fill="gradient",  # Mark as gradient for debugging
            strokeWidth=None
        )
        
        self.elements.append(placeholder)
    
    def paint_path(self, gstate, stroke, fill, evenodd, path):
        """Handle vector path drawing - create placeholder for later hydration"""
        
        # Only create placeholders for visible paths (stroked or filled)
        if not stroke and not fill:
            return
        
        # Track vectors in Form XObjects
        if self._form_xobject_depth > 0:
            self._vectors_in_form_xobjects += 1
        
        # Flush any accumulated text before vector
        if self.current_rotated_segment:
            self._flush_rotated_segment()
        self._flush_text()
        
        # Calculate approximate bounding box from path
        try:
            # Get path bounds
            if not path:
                return
            
            # Extract coordinates from path
            points = []
            for segment_type, *coords in path:
                if segment_type in ('m', 'l'):  # moveto, lineto
                    points.append((coords[0], coords[1]))
                elif segment_type == 'c':  # curveto
                    points.extend([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
                elif segment_type == 're':  # rectangle
                    x, y, w, h = coords
                    points.extend([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
            
            if not points:
                return
            
            # Transform points by CTM AND flip Y coordinate (to match VectorExtractionDevice)
            transformed_points = []
            for p in points:
                tp = apply_matrix_pt(self.current_ctm, p)
                # Flip Y coordinate to top-left origin
                tp = (tp[0], self.page_height - tp[1])
                transformed_points.append(tp)
            
            # Calculate bounding box from FLIPPED points
            xs = [p[0] for p in transformed_points]
            ys = [p[1] for p in transformed_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Bounding box dimensions
            width = x_max - x_min
            height = y_max - y_min
            x = x_min
            y = y_min  # Y is already flipped, so use y_min directly
            
            # Add stroke width padding for stroked paths (match VectorExtractionDevice behavior)
            if stroke:
                stroke_width = getattr(gstate, 'linewidth', 1.0)
                if stroke_width == 0:
                    stroke_width = 1.0  # Default stroke width when not explicitly set
                
                # Expand bbox by stroke_width on all sides
                x -= stroke_width
                y -= stroke_width
                width += stroke_width * 2
                height += stroke_width * 2
            
            # Skip tiny or invalid vectors
            if width < 0.1 or height < 0.1:
                return
            
            # Create vector placeholder
            self.vector_counter += 1
            placeholder = PdfVectorElement(
                id=f"vec_placeholder_{self.page_num}_{self.vector_counter}",
                type="vector",
                x=round(x, COORDINATE_PRECISION),
                y=round(y, COORDINATE_PRECISION),
                width=round(width, COORDINATE_PRECISION),
                height=round(height, COORDINATE_PRECISION),
                svgContent="",  # Will be hydrated later
                pathData="",    # Will be hydrated later
                opacity=1.0,
                stroke=None,
                fill=None,
                strokeWidth=None
            )
            
            # Only append if vector extraction is enabled
            if not self.skip_vector_placeholders:
                self.elements.append(placeholder)
            
        except Exception:
            pass