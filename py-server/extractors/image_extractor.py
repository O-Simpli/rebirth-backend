"""
PDF Image Extraction Module

Image extraction using pdfminer.six for low-level PDF processing.
Features CTM tracking, coordinate transformations, graphics state management,
and comprehensive image decompression pipeline.
"""

import base64
import hashlib
import io
import logging
import zlib
import math
from typing import Dict, List, Optional, Tuple

from PIL import Image

# pdfminer.six imports for low-level PDF processing
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.utils import apply_matrix_pt, mult_matrix

from models.pdf_types import PdfImageElement
from utils.pdf_transforms import Transformation, decompose_ctm, get_image_pivot_point

logger = logging.getLogger(__name__)


# --- Module Constants ---

IMAGE_EXTRACTION_CONSTANTS = {
    'COORDINATE_PRECISION': 2,
    'MIN_IMAGE_SIZE': 1,
    'MAX_IMAGE_SIZE': 10000,
    'DEFAULT_DPI': 150,
}

# --- Core Extraction Device ---

class ImageExtractionDevice(PDFDevice):
    """
    PDF image extraction device with graphics state and transformation tracking.
    
    Processes PDF operator streams to extract images with accurate positioning,
    handling inline images, Form XObjects, and nested graphics state.
    """

    def __init__(self, rsrcmgr: PDFResourceManager, include_image_data: bool = True):
        super().__init__(rsrcmgr)  # Pass rsrcmgr to parent PDFDevice
        self.rsrcmgr = rsrcmgr
        self.include_image_data = include_image_data
        self.images = []
        self.ctm_stack = []
        self.current_ctm = [1, 0, 0, 1, 0, 0]  # Identity matrix [a, b, c, d, e, f]
        self.page_height = 0  # Will be set when processing page

        # Advanced graphics state tracking
        self.graphics_state_stack = []
        self.current_graphics_state = {
            'ctm': [1, 0, 0, 1, 0, 0],
            'clip_path': None,
            'opacity': 1.0,
            'blend_mode': 'Normal',
            'soft_mask': None,
            'stroke_alpha': 1.0,
            'fill_alpha': 1.0
        }

        # Operator context tracking
        self.in_text_object = False
        self.in_form_xobject = False
        self.current_form_name = None
        self.form_stack = []

        # Image processing enhancement
        self.processed_images = set()  # Track already processed images
        self.image_context = {}  # Store context for each image

        # Raw image stream tracking for advanced processing
        self.raw_image_streams = {}

    # --- Page and State Management ---

    def set_ctm(self, ctm):
        """Set the current transformation matrix"""
        self.current_ctm = list(ctm)

    def begin_page(self, page, ctm):
        """Called when a new page begins processing"""
        self.images = []  # Reset images for new page
        self.ctm_stack = []
        self.current_ctm = list(ctm) if ctm else [1, 0, 0, 1, 0, 0]

        # Reset graphics state for new page
        self.graphics_state_stack = []
        self.current_graphics_state = {
            'ctm': list(self.current_ctm),
            'clip_path': None,
            'opacity': 1.0,
            'blend_mode': 'Normal',
            'soft_mask': None,
            'stroke_alpha': 1.0,
            'fill_alpha': 1.0
        }

        # Reset operator context tracking
        self.in_text_object = False
        self.in_form_xobject = False
        self.current_form_name = None
        self.form_stack = []
        self.processed_images.clear()
        self.image_context.clear()

        # Get page height for coordinate conversion
        if hasattr(page, 'height'):
            self.page_height = page.height
        else:
            self.page_height = IMAGE_EXTRACTION_CONSTANTS.get('DEFAULT_PAGE_HEIGHT', 842)

    def end_page(self, page):
        """Called when page processing ends"""
        pass

    def begin_figure(self, name, bbox, matrix):
        """Handle figure/form XObject - may contain images"""
        # Enhanced form tracking
        self.in_form_xobject = True
        self.current_form_name = name
        self.form_stack.append({
            'name': name,
            'bbox': bbox,
            'matrix': matrix,
            'parent_ctm': list(self.current_ctm)
        })

        # Save current CTM and apply figure transformation
        self.ctm_stack.append(self.current_ctm[:])  # Make a copy of the list
        if matrix:
            self.current_ctm = mult_matrix(self.current_ctm, matrix)

    def end_figure(self, name):
        """End figure processing"""
        # Enhanced form stack management
        if self.form_stack:
            self.form_stack.pop()

        if not self.form_stack:
            self.in_form_xobject = False
            self.current_form_name = None
        else:
            # Restore previous form context
            self.current_form_name = self.form_stack[-1]['name']

        # Restore previous CTM
        if self.ctm_stack:
            self.current_ctm = self.ctm_stack.pop()

    # --- Graphics State and Operator Context ---

    def push_graphics_state(self):
        """Save current graphics state (q operator)"""
        state_copy = {
            'ctm': list(self.current_graphics_state['ctm']),
            'clip_path': self.current_graphics_state['clip_path'],
            'opacity': self.current_graphics_state['opacity'],
            'blend_mode': self.current_graphics_state['blend_mode'],
            'soft_mask': self.current_graphics_state['soft_mask'],
            'stroke_alpha': self.current_graphics_state['stroke_alpha'],
            'fill_alpha': self.current_graphics_state['fill_alpha']
        }
        self.graphics_state_stack.append(state_copy)
        self.ctm_stack.append(list(self.current_ctm))

    def pop_graphics_state(self):
        """Restore previous graphics state (Q operator)"""
        if self.graphics_state_stack:
            self.current_graphics_state = self.graphics_state_stack.pop()
            self.current_ctm = self.current_graphics_state['ctm']

        if self.ctm_stack:
            self.ctm_stack.pop()

    def update_ctm(self, matrix):
        """Update current transformation matrix (cm operator)"""
        # Multiply current CTM with the new matrix
        self.current_ctm = mult_matrix(self.current_ctm, matrix)
        self.current_graphics_state['ctm'] = list(self.current_ctm)

    def set_graphics_state(self, state_dict):
        """Set graphics state parameters (gs operator)"""
        for key, value in state_dict.items():
            if key in self.current_graphics_state:
                self.current_graphics_state[key] = value
                logger.debug(f"Set graphics state {key}: {value}")

    def begin_text_object(self):
        """Handle beginning of text object (BT operator)"""
        self.in_text_object = True

    def end_text_object(self):
        """Handle end of text object (ET operator)"""
        self.in_text_object = False
        
    def process_operator(self, tag, args):
        """
        Process specific PDF operators for graphics state tracking.
        
        Operator stream processing for complex PDF structures.
        """
        try:
            if tag == 'q':  # Save graphics state
                self.push_graphics_state()
            elif tag == 'Q':  # Restore graphics state
                self.pop_graphics_state()
            elif tag == 'cm':  # Modify transformation matrix
                if len(args) == 6:
                    self.update_ctm(args)
            elif tag == 'BT':  # Begin text object
                self.begin_text_object()
            elif tag == 'ET':  # End text object
                self.end_text_object()
            elif tag == 'gs':  # Set graphics state from dictionary
                if args and hasattr(args[0], 'resolve'):
                    state_dict = args[0].resolve()
                    self.set_graphics_state(state_dict)
            elif tag in ['BI', 'ID', 'EI']:  # Inline image operators
                if tag == 'BI':
                    # Begin inline image - start collecting image dictionary
                    self._inline_image_dict = {}
                    self._inline_image_state = 'dict'
                elif tag == 'ID':
                    # Image data follows - prepare for data collection
                    self._inline_image_state = 'data'
                elif tag == 'EI':
                    # End inline image - process the collected data
                    if hasattr(self, '_inline_image_dict') and hasattr(self, '_inline_image_data'):
                        self._process_inline_image()
                    self._cleanup_inline_image_state()
                # Note: Full inline image processing requires interpreter-level access
                
        except Exception as e:
            logger.debug(f"Error processing operator {tag}: {e}")

    # --- Unused Required Methods ---

    def begin_tag(self, tag, props=None):
        """Handle tagged content"""
        pass

    def end_tag(self):
        """End tagged content"""
        pass

    def do_tag(self, tag, props=None):
        """Handle single tag"""
        pass

    def paint_path(self, gstate, stroke, fill, evenodd, path):
        """Handle path painting - not relevant for image extraction"""
        pass

    def render_string(self, textstate, seq, ncs, gcs):
        """Handle text rendering - not relevant for image extraction"""
        pass

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, gcs, adv, textobjects):
        """Handle individual character rendering - not relevant for image extraction"""
        pass

    # --- Image Rendering and Processing ---

    def paint_image(self, name, stream, ctm=None):
        """
        Override paint_image to capture raw image streams during PDF processing.
        
        This method is called by pdfminer when processing PDF operators that paint images.
        It captures the raw image stream data before any internal processing.
        
        Args:
            name: Image name/reference
            stream: Raw PDF image stream object
            ctm: Current transformation matrix
        """
        try:
            # Store raw stream reference for later processing
            if hasattr(self, 'raw_image_streams'):
                if not hasattr(self, 'raw_image_streams'):
                    self.raw_image_streams = {}
                self.raw_image_streams[name] = {
                    'stream': stream,
                    'ctm': ctm,
                    'page_context': {
                        'page_height': getattr(self, 'page_height', None),
                        'graphics_state': dict(getattr(self, 'current_graphics_state', {})),
                        'form_context': getattr(self, 'current_form_name', None)
                    }
                }

            # Call the standard render_image for normal processing
            return self.render_image(name, stream)

        except Exception as e:
            logger.error(f"Error in paint_image for {name}: {e}")
            # Fallback to standard processing
            return self.render_image(name, stream)

    def render_image(self, name, stream):
        """
        Handle image rendering using common image processing logic.
        
        Primary entry point for image extraction with comprehensive context detection.
        """
        try:
            logger.debug(f"Rendering image: {name}")

            # Conditionally extract image data based on include_image_data flag
            if self.include_image_data:
                # Extract image data (expensive operation)
                image_data, image_format, width, height = self._convert_stream_to_image(stream)
            else:
                # Extract only basic metadata without processing image data
                width, height, image_format = self._extract_basic_image_metadata(stream)
                image_data = None

            if width <= 0 or height <= 0:
                logger.debug(f"Skipping image {name} with invalid dimensions: {width}x{height}")
                return

            # Use common processing logic
            self._process_image_common(name, width, height, image_data, image_format, width, height)

        except Exception as e:
            logger.error(f"Error rendering image {name}: {e}")

    def _process_image_common(self, name: str, width: int, height: int, image_data: Optional[bytes] = None,
                            image_format: Optional[str] = None, orig_width: int = 0, orig_height: int = 0) -> bool:
        """Common image processing logic for both regular and inline images."""
        try:
            # Calculate normalized (pre-transformation) position and dimensions
            transformation = decompose_ctm(self.current_ctm)
            untransformed_width = abs(transformation.scaleX)
            untransformed_height = abs(transformation.scaleY)

            # The untransformed position is the translation component of the CTM.
            bottom_left_x = transformation.translateX
            bottom_left_y = transformation.translateY

            # The untransformed size is the scaling component.
            img_width = untransformed_width
            img_height = untransformed_height

            anchor_x, anchor_y = get_image_pivot_point(
                bottom_left_x, bottom_left_y, img_height, transformation.rotation
            )

            # Validate image size using PDF coordinates
            if not self._validate_image_dimensions(img_width, img_height, f"{name} "):
                return False

            # Generate stable ID using helper function
            precision = IMAGE_EXTRACTION_CONSTANTS['COORDINATE_PRECISION']
            image_id = generate_image_id(
                name=name,
                x=round(anchor_x, precision),
                y=round(anchor_y, precision),
                width=round(img_width, precision),
                height=round(img_height, precision)
            )

            # Check for duplicate processing
            if image_id in self.processed_images:
                logger.debug(f"Skipping duplicate image: {name}")
                return False

            # If extraction failed but we have basic dimensions, still track the image
            if orig_width == 0 or orig_height == 0:
                orig_width = width
                orig_height = height
                    
            # Set default format if none provided
            if image_format is None:
                image_format = 'PNG'

            # Enhanced context tracking
            context_info = {
                'in_form': self.in_form_xobject,
                'form_name': self.current_form_name,
                'form_stack_depth': len(self.form_stack),
                'graphics_state': dict(self.current_graphics_state),
                'in_text_object': self.in_text_object
            }

            image_info = self._create_base_image_info(
                name,
                screen_x=anchor_x,
                screen_y=anchor_y,
                screen_width=img_width,
                screen_height=img_height,
                original_width=orig_width,
                original_height=orig_height,
                image_format=image_format,
                image_data=image_data,
                x=bottom_left_x,
                y=bottom_left_y,
                img_width=img_width,
                img_height=img_height,
                context_info=context_info,
                image_id=image_id,
                transformation=transformation
            )

            # Store in context tracking
            self.processed_images.add(image_id)
            self.image_context[name] = context_info

            # Apply graphics state processing
            processed_image_info = self._apply_graphics_state_to_image(image_info)

            self.images.append(processed_image_info)
            logger.debug(f"Processed image {name}: {img_width:.1f}x{img_height:.1f} at PDF coords ({anchor_x:.1f}, {anchor_y:.1f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in common image processing for {name}: {e}")
            return False
    
    def handle_inline_image(self, width, height, image_data, filter_params=None):
        """
        Handle inline images using common image processing logic.
        These are images embedded directly in the content stream.
        """
        try:
            logger.debug(f"Processing inline image: {width}x{height}")

            # Generate unique name for inline image
            inline_name = f"inline_image_{len(self.images)}"

            # Conditionally process image data based on include_image_data flag
            if not self.include_image_data:
                image_data = None  # Don't process the image data blob

            # Use common processing logic
            self._process_image_common(inline_name, width, height, image_data, 'PNG', width, height)
                
        except Exception as e:
            logger.error(f"Error handling inline image: {e}")
            
    def _process_inline_image(self):
        """Process collected inline image data from BI...ID...EI operators."""
        try:
            # Extract image properties from dictionary
            width = self._inline_image_dict.get('W', self._inline_image_dict.get('Width', 0))
            height = self._inline_image_dict.get('H', self._inline_image_dict.get('Height', 0))
            bits_per_component = self._inline_image_dict.get('BPC', 
                                     self._inline_image_dict.get('BitsPerComponent', 8))
            color_space = self._inline_image_dict.get('CS', 
                                  self._inline_image_dict.get('ColorSpace', 'DeviceRGB'))
            
            if width <= 0 or height <= 0:
                logger.debug(f"Invalid inline image dimensions: {width}x{height}")
                return
                
            # Process any filters
            filters = self._inline_image_dict.get('F', self._inline_image_dict.get('Filter', []))
            if not isinstance(filters, list):
                filters = [filters] if filters else []
                
            # Get raw image data
            raw_data = getattr(self, '_inline_image_data', b'')
            
            # Decompress if needed
            if filters:
                decompressed_data = self.decompress_image_stream(raw_data, filters)
                if decompressed_data is None:
                    logger.debug("Failed to decompress inline image")
                    return
            else:
                decompressed_data = raw_data
                
            # Create a mock stream object to use the unified pipeline
            mock_stream = self._create_mock_stream(width, height, color_space, bits_per_component, decompressed_data)
            
            # Use the unified conversion pipeline
            image_bytes, format_type, proc_width, proc_height = self._convert_stream_to_image(mock_stream)
            
            if image_bytes:
                # Call the existing inline image handler with processed data
                self.handle_inline_image(width, height, image_bytes)
            else:
                logger.debug("Failed to process inline image data via unified pipeline")
                
        except Exception as e:
            logger.error(f"Error processing inline image: {e}")

    def _cleanup_inline_image_state(self):
        """Clean up inline image processing state"""
        if hasattr(self, '_inline_image_dict'):
            delattr(self, '_inline_image_dict')
        if hasattr(self, '_inline_image_data'):
            delattr(self, '_inline_image_data')
        if hasattr(self, '_inline_image_state'):
            delattr(self, '_inline_image_state')

    # --- Image Attribute and Metadata Helpers ---

    def _apply_graphics_state_to_image(self, image_info):
        """Apply current graphics state properties to enhance image extraction."""
        try:
            # Create enhanced image info with graphics state data
            enhanced_info = image_info.copy()
            
            # Apply opacity/transparency
            opacity = self.current_graphics_state.get('opacity', 1.0)
            fill_alpha = self.current_graphics_state.get('fill_alpha', 1.0)
            stroke_alpha = self.current_graphics_state.get('stroke_alpha', 1.0)
            
            # Calculate effective transparency
            effective_alpha = min(opacity, fill_alpha)
            
            # Add transparency information
            enhanced_info['transparency'] = {
                'opacity': opacity,
                'fill_alpha': fill_alpha,
                'stroke_alpha': stroke_alpha,
                'effective_alpha': effective_alpha,
                'is_transparent': effective_alpha < 1.0
            }
            
            # Apply blend mode information
            blend_mode = self.current_graphics_state.get('blend_mode', 'Normal')
            enhanced_info['blend_mode'] = blend_mode
            enhanced_info['has_special_blend'] = blend_mode != 'Normal'
            
            # Apply soft mask information
            soft_mask = self.current_graphics_state.get('soft_mask')
            if soft_mask:
                enhanced_info['soft_mask'] = soft_mask
                enhanced_info['has_mask'] = True
            else:
                enhanced_info['has_mask'] = False
                
            # Apply clipping path information
            clip_path = self.current_graphics_state.get('clip_path')
            if clip_path:
                enhanced_info['clipping'] = {
                    'has_clip': True,
                    'clip_path': clip_path,
                    'potentially_clipped': True
                }
                enhanced_info['extraction_quality'] = 'potentially_clipped'
            else:
                enhanced_info['clipping'] = {'has_clip': False}
                enhanced_info['extraction_quality'] = 'full'
                
            # Add processing metadata
            enhanced_info['graphics_state_applied'] = True
            enhanced_info['processing_flags'] = []
            
            if effective_alpha < 1.0:
                enhanced_info['processing_flags'].append('transparent')
            if blend_mode != 'Normal':
                enhanced_info['processing_flags'].append('special_blend')
            if soft_mask:
                enhanced_info['processing_flags'].append('masked')
            if clip_path:
                enhanced_info['processing_flags'].append('clipped')
                
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Error applying graphics state to image: {e}")
            return image_info  # Return original if processing fails

    def calculate_image_position(self, image_ctm) -> Tuple[float, float, float, float, Transformation]:
        """
        Calculate precise image position and transformation data using CTM.
        
        In PDF, images are placed in a unit square (0,0) to (1,1) and then 
        transformed by the CTM. The CTM format is [a, b, c, d, e, f] representing:
        | a  c  e |
        | b  d  f |
        | 0  0  1 |
        
        Returns tuple of (x, y, width, height, transformation) in PDF coordinates (bottom-left origin).
        Client will handle screen coordinate conversion based on rendering context.
        """
        # Image is placed in unit coordinates (0,0) to (1,1)
        # Transform the unit square corners using the current CTM
        unit_corners = [
            (0, 0),    # bottom-left
            (1, 0),    # bottom-right  
            (1, 1),    # top-right
            (0, 1)     # top-left
        ]
        
        # Apply the current CTM to transform the unit square
        transformed_corners = [
            apply_matrix_pt(image_ctm, corner) 
            for corner in unit_corners
        ]
        
        # Calculate bounding box from transformed corners
        x_coords = [corner[0] for corner in transformed_corners]
        y_coords = [corner[1] for corner in transformed_corners]
        
        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Ensure positive dimensions
        width = abs(width)
        height = abs(height)
        
        # Decompose CTM to get transformation details
        transformation = decompose_ctm(image_ctm)
        
        return x, y, width, height, transformation

    def _validate_image_dimensions(self, screen_width, screen_height, image_type=""):
        min_size = IMAGE_EXTRACTION_CONSTANTS['MIN_IMAGE_SIZE']
        max_size = IMAGE_EXTRACTION_CONSTANTS['MAX_IMAGE_SIZE']
        
        # Check minimum size
        if screen_width < min_size or screen_height < min_size:
            return False
            
        # Check maximum size
        if screen_width > max_size or screen_height > max_size:
            return False
            
        return True

    def _create_base_image_info(self, name, screen_x, screen_y, screen_width, screen_height, 
                                original_width, original_height, image_format, image_data, 
                                x, y, img_width, img_height, context_info, image_id, transformation):
        precision = IMAGE_EXTRACTION_CONSTANTS['COORDINATE_PRECISION']
        
        return {
            'id': image_id,
            'name': name,
            'index': len(self.images),
            'x': round(screen_x, precision),
            'y': round(screen_y, precision),
            'width': round(screen_width, precision),
            'height': round(screen_height, precision),
            'original_width': original_width,
            'original_height': original_height,
            'format': image_format,
            'data': image_data,
            'ctm': self.current_ctm[:],  # Make a copy
            'pdf_coords': (x, y, img_width, img_height),
            'context': context_info,
            'rotation': round(transformation.rotation, precision),
            'skewX': round(transformation.skewX, precision),
            'skewY': round(transformation.skewY, precision)
        }

    # --- Data Decompression and Conversion ---

    def _convert_stream_to_image(self, stream_obj) -> tuple[Optional[bytes], Optional[str], int, int]:
        """
        Unified pipeline to process a raw PDF image stream into a standard image format.
        """
        try:
            # Get image properties
            width = stream_obj.get('Width', 0)
            height = stream_obj.get('Height', 0)
            if width <= 0 or height <= 0:
                return None, None, 0, 0
            
            colorspace = stream_obj.get('ColorSpace')
            bits_per_component = stream_obj.get('BitsPerComponent', 8)
            filters = stream_obj.get('Filter', [])
            if not isinstance(filters, list):
                filters = [filters] if filters else []

            # Get raw data and decompress it
            raw_data = stream_obj.get_rawdata()
            decompressed_data = self.decompress_image_stream(raw_data, filters) if filters else raw_data
            if decompressed_data is None:
                return None, None, width, height

            # Extract ICC Profile if it exists
            icc_profile = None
            if isinstance(colorspace, list) and hasattr(colorspace[0], 'name') and colorspace[0].name == 'ICCBased':
                try:
                    icc_stream = colorspace[1].resolve()
                    icc_profile = icc_stream.get_data()
                except Exception:
                    pass

            # Determine if the image is a JPEG by its filter
            filter_names = [f.name if hasattr(f, 'name') else str(f) for f in filters]
            is_jpeg = any(f in ['DCTDecode', 'DCT'] for f in filter_names)

            if is_jpeg:
                # --- HIGH-PERFORMANCE JPEG PATH ---
                try:
                    buffer = io.BytesIO()
                    img = Image.open(io.BytesIO(decompressed_data))
                    # For JPEGs, if there's a soft mask, we must convert to PNG to support transparency.
                    if stream_obj.get('SMask'):
                        pass # Fall through to PNG conversion
                    else:
                        img.save(buffer, format='JPEG', quality=95, icc_profile=icc_profile)
                        image_bytes = buffer.getvalue()
                        return image_bytes, "JPEG", width, height
                except Exception:
                    pass # Fall back to PNG

            # --- DEFAULT PNG CONVERSION PATH ---
            base_pil_image = None
            if is_jpeg:
                base_pil_image = Image.open(io.BytesIO(decompressed_data))
            else:
                base_mode = self._get_pil_mode(stream_obj)
                components = self.get_components_per_pixel(colorspace)
                expected_size = width * height * components * bits_per_component // 8
                processed_data = decompressed_data
                if len(processed_data) < expected_size:
                    processed_data += b'\x00' * (expected_size - len(processed_data))
                base_pil_image = self._create_pil_image_from_data(processed_data, width, height, base_mode, bits_per_component, return_pil_image=True)

            if not base_pil_image:
                return None, None, width, height
            
            # --- Common Post-Processing for All Images Being Converted to PNG ---
            
            # Apply transparency if present
            final_pil_image = base_pil_image
            smask = stream_obj.get('SMask')
            if smask:
                try:
                    smask_stream = smask.resolve()
                    if smask_stream:
                        smask_width = smask_stream.get('Width', 0)
                        smask_height = smask_stream.get('Height', 0)
                        smask_bits = smask_stream.get('BitsPerComponent', 8)
                        smask_raw_data = smask_stream.get_data() # get_data() is safer
                        
                        if smask_raw_data and smask_width > 0 and smask_height > 0:
                            alpha_mask = self._create_pil_image_from_data(smask_raw_data, smask_width, smask_height, 'L', smask_bits, return_pil_image=True)
                            if alpha_mask:
                                if final_pil_image.mode != 'RGBA':
                                    final_pil_image = final_pil_image.convert('RGBA')
                                
                                if alpha_mask.size != final_pil_image.size:
                                    alpha_mask = alpha_mask.resize(final_pil_image.size, Image.Resampling.LANCZOS)
                                    
                                final_pil_image.putalpha(alpha_mask)
                except Exception:
                    pass
            
            # Convert to PNG, embedding profile for correct colors
            buffer = io.BytesIO()
            final_pil_image.save(buffer, format='PNG', optimize=True, icc_profile=icc_profile)
            image_bytes = buffer.getvalue()
            
            return image_bytes, "PNG", width, height

        except Exception as e:
            logger.error(f"CRITICAL error in _convert_stream_to_image: {e}", exc_info=True)
            return None, None, 0, 0
            
    def _extract_basic_image_metadata(self, obj):
        """Extract basic image metadata (dimensions and format) without processing image data."""
        try:
            # Get basic stream properties
            if hasattr(obj, 'attrs'):
                attrs = obj.attrs
                width = int(attrs.get('Width', 0))
                height = int(attrs.get('Height', 0))
                
                # Determine format from filters without decompressing
                filters = attrs.get('Filter', [])
                if not isinstance(filters, list):
                    filters = [filters] if filters else []
                
                image_format = self.determine_image_format(filters)
                
                return width, height, image_format
            else:
                # Fallback for objects without attributes
                return 0, 0, 'PNG'
                
        except Exception:
            return 0, 0, 'PNG'

    def decompress_image_stream(self, raw_data: bytes, filters: List) -> Optional[bytes]:
        """
        Decompress image stream data based on PDF filters.
        """
        try:
            # Define filter mappings - more maintainable and extensible
            filter_map = {
                'FlateDecode': self.decompress_flate,
                'Fl': self.decompress_flate,
                'DCTDecode': lambda data: data,  # JPEG - already compressed
                'DCT': lambda data: data,
                'CCITTFaxDecode': lambda data: data,  # Not implemented yet
                'CCF': lambda data: data,
                'LZWDecode': lambda data: data,  # Not implemented yet
                'LZW': lambda data: data,
                'RunLengthDecode': self.decompress_runlength,
                'RL': self.decompress_runlength,
                'ASCII85Decode': self.decode_ascii85,
                'A85': self.decode_ascii85,
                'ASCIIHexDecode': self.decode_ascii_hex,
                'AHx': self.decode_ascii_hex,
            }
            
            current_data = raw_data
            
            # Process filters in reverse order (PDF spec requirement)
            for filter_obj in reversed(filters):
                filter_name = str(filter_obj).replace('/', '').replace("'", "")
                
                if filter_name in filter_map:
                    decompress_func = filter_map[filter_name]
                    current_data = decompress_func(current_data)
                    
                    if current_data is None:
                        return None
                else:
                    logger.debug(f"Unknown filter: {filter_name}")
                    
            return current_data
            
        except Exception as e:
            logger.error(f"Error in decompress_image_stream: {e}")
            return None

    def decompress_flate(self, data: bytes) -> Optional[bytes]:
        """Decompress FlateDecode (zlib) compressed data with systematic fallback methods"""
        if not data:
            return None
        
        # Define decompression strategies to try in order
        strategies = [
            # Method 1: Standard zlib decompression
            lambda: zlib.decompress(data),
            # Method 2: Raw deflate without zlib header
            lambda: zlib.decompress(data, -15),
        ]
        
        # Method 3: Try skipping potential extra bytes at the beginning
        for skip in [1, 2, 3, 4]:
            if len(data) > skip:
                strategies.extend([
                    lambda s=skip: zlib.decompress(data[s:]),
                    lambda s=skip: zlib.decompress(data[s:], -15),
                ])
        
        # Try each strategy until one succeeds
        for strategy in strategies:
            try:
                result = strategy()
                return result
            except zlib.error:
                continue
                
        return None

    def decompress_runlength(self, data: bytes) -> Optional[bytes]:
        """Decompress Run-length encoded data"""
        try:
            result = bytearray()
            i = 0
            while i < len(data):
                length = data[i]
                if length == 128:  # EOD marker
                    break
                elif length < 128:
                    # Copy next length+1 bytes literally
                    count = length + 1
                    result.extend(data[i+1:i+1+count])
                    i += count + 1
                else:
                    # Repeat next byte 257-length times
                    count = 257 - length
                    if i + 1 < len(data):
                        result.extend([data[i+1]] * count)
                    i += 2
            return bytes(result)
        except Exception as e:
            logger.error(f"RunLength decompression failed: {e}")
            return None

    def decode_ascii85(self, data: bytes) -> Optional[bytes]:
        """Decode ASCII85 encoded data"""
        # This implementation is a simplified one. For full spec compliance,
        # a more robust library might be needed.
        try:
            return base64.a85decode(data, adobe=True)
        except Exception as e:
            logger.error(f"ASCII85 decode failed: {e}")
            return None

    def decode_ascii_hex(self, data: bytes) -> Optional[bytes]:
        """Decode ASCII hex encoded data"""
        try:
            # Remove the EOD marker '>' before decoding
            cleaned_data = data.rstrip(b'>')
            return base64.b16decode(cleaned_data, casefold=True)
        except Exception as e:
            logger.error(f"ASCII hex decode failed: {e}")
            return None

    # --- Data and Color Space Helpers ---

    def _get_pil_mode(self, stream_obj) -> str:
        """
        Determine the PIL image mode from the PDF stream's color space.
        Handles simple, Indexed, and ICCBased color spaces consistently.
        """
        color_space = stream_obj.get('ColorSpace')
        
        if color_space is None:
            return 'RGB'

        # Handle simple color spaces like /DeviceRGB
        if not isinstance(color_space, list):
            cs_str = str(color_space).replace('/', '')
            if cs_str in ['DeviceRGB', 'RGB', 'CalRGB']:
                return 'RGB'
            if cs_str in ['DeviceGray', 'Gray', 'G', 'CalGray']:
                return 'L'
            if cs_str in ['DeviceCMYK', 'CMYK']:
                return 'CMYK'
            return 'RGB'

        # Handle complex, list-based color spaces like [/ICCBased, <...>]
        try:
            cs_name_obj = color_space[0]
            
            # Robustly get the name of the colorspace from the pdfminer.six object.
            if hasattr(cs_name_obj, 'name'):
                cs_type = cs_name_obj.name
            elif isinstance(cs_name_obj, bytes):
                cs_type = cs_name_obj.decode('utf-8')
            else:
                cs_type = str(cs_name_obj).replace('/', '').strip("'\"")

            if cs_type == 'Indexed':
                return 'P'

            if cs_type == 'ICCBased':
                # For ICCBased, the underlying mode is determined by the number of components ('N')
                icc_profile_stream = color_space[1].resolve()
                num_components = icc_profile_stream.get('N')
                
                if num_components == 1:
                    return 'L'
                if num_components == 3:
                    return 'RGB'
                if num_components == 4:
                    return 'CMYK'
                
                return 'RGB'
            
            return 'RGB'

        except Exception:
            return 'RGB'
    def get_components_per_pixel(self, color_space: str) -> int:
        """Get number of color components per pixel for given color space"""
        color_space_map = {
            'DeviceRGB': 3, 'RGB': 3,
            'DeviceGray': 1, 'Gray': 1, 'G': 1,
            'DeviceCMYK': 4, 'CMYK': 4,
            'Indexed': 1,  # Indexed color spaces use palette lookup
            'CalRGB': 3, 'CalGray': 1,  # Calibrated color spaces
            'Lab': 3, 'ICCBased': 3,  # Advanced color spaces
        }
        # Handle complex color space objects
        if hasattr(color_space, '__getitem__'):
            try:
                cs_name = str(color_space[0]).replace('/', '') if color_space else 'DeviceRGB'
                return color_space_map.get(cs_name, 3)
            except:
                pass
        
        cs_str = str(color_space).replace('/', '').replace("'", "") if color_space else 'DeviceRGB'
        return color_space_map.get(cs_str, 3)  # Default to RGB

    def _create_pil_image_from_data(self, data: bytes, width: int, height: int, mode: str, bits_per_component: int = 8, return_pil_image: bool = False):
        """Generic PIL image creation and conversion from raw pixel data."""
        try:
            # Handle different bit depths
            processed_data = data
            if bits_per_component != 8:
                if bits_per_component == 16:
                    processed_data = bytes(data[i] for i in range(1, len(data), 2))
                elif bits_per_component < 8:
                    pixels_per_byte = 8 // bits_per_component
                    unpacked_data = bytearray()
                    mask = (1 << bits_per_component) - 1
                    scale = 255 // mask
                    for byte in data:
                        for i in range(pixels_per_byte):
                            shift = (pixels_per_byte - 1 - i) * bits_per_component
                            pixel = (byte >> shift) & mask
                            unpacked_data.append(pixel * scale)
                    processed_data = bytes(unpacked_data)
            
            # Create PIL image from raw bytes
            img = Image.frombytes(mode, (width, height), processed_data)
            
            # Return PIL Image object or PNG bytes based on parameter
            if return_pil_image:
                return img
            else:
                # Save to PNG format
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                result = buffer.getvalue()
                
                return result
            
        except Exception as e:
            logger.error(f"Error creating PIL image ({mode}, {width}x{height}): {e}")
            return None

    def determine_image_format(self, filters: List) -> str:
        """Determine image format based on PDF filters and stream properties."""
        try:
            # Check filters for format hints
            for filter_obj in filters:
                filter_name = str(filter_obj).replace('/', '').replace("'", "")
                if filter_name in ['DCTDecode', 'DCT']:
                    return "JPEG"
                elif filter_name in ['JPXDecode']:
                    return "JPEG2000"
                elif filter_name in ['CCITTFaxDecode', 'CCF']:
                    return "TIFF"
            
            # Default format for processed images
            return "PNG"
            
        except Exception as e:
            logger.error(f"Error determining image format: {e}")
            return "PNG"  # Safe default

    def _create_mock_stream(self, width: int, height: int, color_space, bits_per_component: int, decompressed_data: bytes):
        """Create a mock stream object that mimics a PDF stream for use with the unified pipeline."""
        class MockStream:
            def __init__(self, properties, data):
                self._properties = properties
                self._data = data
                
            def get(self, key, default=None):
                return self._properties.get(key, default)
                
            def get_rawdata(self):
                return self._data
                
            def get_data(self):
                return self._data
        
        return MockStream({
            'Width': width,
            'Height': height,
            'ColorSpace': color_space,
            'BitsPerComponent': bits_per_component,
            'Filter': [],  # Inline images are already decompressed
            'SMask': None,  # Inline images don't typically have SMask
            'Mask': None,   # Inline images don't typically have Mask
        }, decompressed_data)

# --- Document Processor Wrapper ---

class PDFDocumentProcessor:
    """High-level PDF document processor with comprehensive resource management."""
    
    def __init__(self, file_path: str, include_image_data: bool = True):
        self.file_path = file_path
        self.include_image_data = include_image_data
        self.fp = None
        self.parser = None
        self.document = None
        self.rsrcmgr = None
        self.device = None
        self.interpreter = None
        
    def __enter__(self):
        """Context manager entry - open and initialize PDF document"""
        try:
            self.fp = open(self.file_path, 'rb')
            self.parser = PDFParser(self.fp)
            self.document = PDFDocument(self.parser)
            
            if not self.document.is_extractable:
                raise ValueError("PDF document is not extractable (may be encrypted)")
            
            self.rsrcmgr = PDFResourceManager()
            self.device = ImageExtractionDevice(self.rsrcmgr, self.include_image_data)
            self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
            
            return self
            
        except Exception as e:
            self.cleanup()
            raise ValueError(f"Failed to initialize PDF document: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up all resources"""
        if self.fp:
            try:
                self.fp.close()
            except Exception:
                pass
        self.fp = None
        self.parser = None
        self.document = None
        self.rsrcmgr = None
        self.device = None
        self.interpreter = None
        
    def extract_images_from_page(self, page_number: int) -> tuple[List[Dict], float, float]:
        """Extract images from a specific page using low-level PDF processing."""
        if not self.document or not self.interpreter:
            raise ValueError("Document processor not initialized")
            
        try:
            pages = list(PDFPage.create_pages(self.document))
            
            if page_number < 1 or page_number > len(pages):
                raise ValueError(f"Page {page_number} out of range (1-{len(pages)})")
            
            page = pages[page_number - 1]
            
            # Extract page dimensions
            page_width, page_height = 612.0, 792.0 # Default
            if hasattr(page, 'mediabox'):
                mediabox = page.mediabox
                page_width = float(mediabox[2] - mediabox[0])
                page_height = float(mediabox[3] - mediabox[1])
            
            # Process the page
            self.interpreter.process_page(page)
            
            # Get images from our custom device
            if self.device and hasattr(self.device, 'images'):
                return self.device.images, page_width, page_height
            else:
                return [], page_width, page_height
                
        except Exception as e:
            logger.error(f"Error extracting images from page {page_number}: {e}")
            return [], 612.0, 792.0


# --- Module-level Utility Functions ---

def generate_image_id(name: str = "unknown", 
                      x: float = 0, 
                      y: float = 0, 
                      width: float = 0, 
                      height: float = 0) -> str:
    """Generate stable ID for image using SHA-256 hash."""
    signature = f"{name}_{x}_{y}_{width}_{height}"
    return hashlib.sha256(signature.encode('utf-8')).hexdigest()

def detect_image_mime_type(img_bytes: bytes) -> str:
    """Detect MIME type from image bytes."""
    try:
        if not img_bytes or len(img_bytes) < 8:
            return "image/unknown"
        
        if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif img_bytes[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
            return "image/webp"
        
        # Fallback to PIL
        img = Image.open(io.BytesIO(img_bytes))
        format_name = img.format.lower() if img.format else 'unknown'
        return f"image/{format_name}"
            
    except Exception:
        return "image/unknown"


# --- Public API Functions ---

def extract_images(file_path: str, start_page: int = 1, end_page: Optional[int] = None, include_image_data: bool = True):
    """Extract all image metadata from a range of PDF pages with accurate positioning, using pdfminer.six."""
    try:
        with PDFDocumentProcessor(file_path, include_image_data) as processor:
            # Determine total number of pages
            if not processor.document:
                logger.error("Failed to open PDF document")
                return []
            
            pages = list(PDFPage.create_pages(processor.document))
            total_pages = len(pages)
            
            # Resolve end_page
            if end_page is None:
                end_page = total_pages
            
            # Validate page range
            if start_page < 1 or start_page > total_pages:
                logger.error(f"Invalid start_page {start_page} (total pages: {total_pages})")
                return []
            
            if end_page < start_page or end_page > total_pages:
                logger.error(f"Invalid end_page {end_page} (start_page: {start_page}, total pages: {total_pages})")
                return []
            
            logger.info(f"Extracting images from pages {start_page} to {end_page} of {total_pages}")
            
            # Process each page in the range
            all_pages_data = []
            
            for page_num in range(start_page, end_page + 1):
                try:
                    # Extract images and page dimensions for this page
                    raw_images, page_width, page_height = processor.extract_images_from_page(page_num)
                    
                    # Convert raw image info to PdfImageElement objects
                    image_elements = []
                    for img_info in raw_images:
                        try:
                            # Convert PDF coordinates (bottom-left origin) to screen coordinates (top-left origin)
                            pdf_x = img_info.get('x', 0)
                            pdf_y = img_info.get('y', 0)
                            width = img_info.get('width', 0)
                            height = img_info.get('height', 0)
                            screen_y = page_height - pdf_y
                            
                            # Prepare image data as data URI if available
                            image_data_uri = None
                            if img_info.get('data'):
                                mime_type = detect_image_mime_type(img_info['data'])
                                base64_data = base64.b64encode(img_info['data']).decode('utf-8')
                                image_data_uri = f"data:{mime_type};base64,{base64_data}"
                            
                            image_elements.append(PdfImageElement(
                                id=img_info.get('id'),
                                type="image",
                                name=img_info.get('name', f'Image_{img_info.get("index", 0)}'),
                                imageIndex=img_info.get('index', 0),
                                x=pdf_x,
                                y=screen_y,
                                width=width,
                                height=height,
                                rotation=-img_info.get('rotation', 0.0),
                                skewX=-img_info.get('skewX', 0.0),
                                skewY=-img_info.get('skewY', 0.0),
                                mimeType=detect_image_mime_type(img_info['data']) if img_info.get('data') else f"image/{img_info.get('format', 'unknown').lower()}",
                                format=img_info.get('format', 'PNG'),
                                data=image_data_uri
                            ))
                        except Exception as e:
                            logger.warning(f"Error converting raw image info to data model on page {page_num}: {e}")
                            continue
                    
                    # Append the list of image elements for this page
                    all_pages_data.append(image_elements)
                    logger.debug(f"Processed page {page_num}: {len(image_elements)} images, {page_width}x{page_height}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    # Continue processing other pages even if one fails
                    continue
            
            logger.info(f"Successfully extracted images from {len(all_pages_data)} pages")
            return all_pages_data
            
    except Exception as e:
        logger.error(f"Error in extract_images: {e}", exc_info=True)
        return [] 
   
def get_image(file_path: str, page_number: int, image_id: str) -> Optional[bytes]:
    """Extract specific image data from a PDF page by ID."""
    try:
        with PDFDocumentProcessor(file_path, include_image_data=True) as processor:
            raw_images, _, _ = processor.extract_images_from_page(page_number)
            
            if not raw_images:
                logger.warning(f"No images found on page {page_number}")
                return None
            
            # Find image by ID
            for img_info in raw_images:
                if img_info.get('id') == image_id:
                    return img_info.get('data')
            
            logger.warning(f"Image with ID {image_id} not found on page {page_number}")
            return None
                
    except Exception as e:
        logger.error(f"Error extracting image {image_id} from page {page_number}: {e}")
        return None