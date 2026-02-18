"""Image Processor for PDFEngine

Handles image extraction operations including placeholder detection,
data extraction, format conversion, and coordinate calculation.
"""

import base64
import io
import logging
from typing import Dict, List, Optional, Tuple, Union

from engine.base_processor import BaseProcessor
from engine.config import ProcessorOptions
from models.pdf_types import PdfImageElement, ProcessingTextRun

logger = logging.getLogger(__name__)


class ImageProcessorOptions(ProcessorOptions):
    """Configuration options for image processing"""
    
    def __init__(
        self,
        include_image_data: bool = True,
        convert_to_png: bool = True,
        max_image_size_mb: float = 10.0,
        compression_quality: int = 95
    ):
        """
        Initialize image processor options.
        
        Args:
            include_image_data: Whether to extract actual image bytes
            convert_to_png: Whether to convert all images to PNG format
            max_image_size_mb: Maximum size for individual images
            compression_quality: JPEG/PNG compression quality (1-100)
        """
        self.include_image_data = include_image_data
        self.convert_to_png = convert_to_png
        self.max_image_size_mb = max_image_size_mb
        self.compression_quality = compression_quality
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'include_image_data': self.include_image_data,
            'convert_to_png': self.convert_to_png,
            'max_image_size_mb': self.max_image_size_mb,
            'compression_quality': self.compression_quality,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageProcessorOptions':
        """Create from dictionary representation"""
        return cls(
            include_image_data=data.get('include_image_data', True),
            convert_to_png=data.get('convert_to_png', True),
            max_image_size_mb=data.get('max_image_size_mb', 10.0),
            compression_quality=data.get('compression_quality', 95),
        )


class ImageProcessor(BaseProcessor):
    """
    Image extraction processor for PDFEngine.
    
    Handles all image extraction operations including detecting images
    in the PDF stream and extracting their actual data.
    """
    
    def __init__(self, engine: 'PDFEngine', options: Optional[ImageProcessorOptions] = None):
        """
        Initialize image processor.
        
        Args:
            engine: Parent PDFEngine instance
            options: ImageProcessorOptions or None for defaults
        """
        super().__init__(engine)
        self.options = options or ImageProcessorOptions()
        
        # Cache for extracted images
        self._image_cache: Dict[str, Tuple[str, str]] = {}
    
    def initialize(self) -> None:
        """Initialize image processor resources"""
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up image processor resources"""
        self._image_cache.clear()
        self._initialized = False
    
    def validate_state(self) -> bool:
        """Validate processor is in valid state"""
        if not self._initialized:
            logger.error("ImageProcessor not initialized")
            return False
        return True
    
    def hydrate_image_data(self, page_index: int, image_name: str) -> Optional[Tuple[str, str]]:
        """
        Extract image data using pikepdf for a given image name on a page.
        
        This separates the "Where" (from pdfminer stream order) from the "What" 
        (actual image bytes via pikepdf).
        
        Args:
            page_index: 0-based page index
            image_name: Name of the image XObject (e.g., "TestImage" or "Im1")
        
        Returns:
            Tuple of (data_uri, mime_type) or None if extraction fails
        """
        # Check cache first
        cache_key = f"{page_index}:{image_name}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        try:
            pikepdf_doc = self.engine.pikepdf_document
            page = pikepdf_doc.pages[page_index]
            
            if not hasattr(page, 'Resources'):
                return None
            
            resources = page.Resources
            image_obj = None
            
            # Try to find the image in direct XObject resources
            if '/XObject' in resources:
                xobjects = resources.XObject
                
                # Try to find the image by name (with and without leading slash)
                for name_variant in [f'/{image_name}', image_name]:
                    if name_variant in xobjects:
                        image_obj = xobjects[name_variant]
                        break
            
            # If not found in direct XObjects, check Pattern resources
            if not image_obj and '/Pattern' in resources:
                patterns = resources.Pattern
                
                for pattern_name, pattern_obj in patterns.items():
                    # Check if pattern has Resources with XObject
                    if '/Resources' in pattern_obj and '/XObject' in pattern_obj.Resources:
                        pattern_xobjects = pattern_obj.Resources.XObject
                        
                        # Try to find the image in pattern's XObjects
                        for name_variant in [f'/{image_name}', image_name]:
                            if name_variant in pattern_xobjects:
                                image_obj = pattern_xobjects[name_variant]
                                break
                        
                        if image_obj:
                            break
            
            if not image_obj:
                return None
            
            # Verify it's an image
            if image_obj.get('/Subtype') != '/Image':
                return None
            
            # Extract image using pikepdf's robust extraction
            try:
                from pikepdf import PdfImage
                
                try:
                    pdf_image = PdfImage(image_obj)
                    
                    # Extract as PIL Image
                    pil_image = pdf_image.as_pil_image()
                    
                    # Check image size
                    image_size_mb = (pil_image.width * pil_image.height * 4) / (1024 * 1024)  # Rough estimate
                    if image_size_mb > self.options.max_image_size_mb:
                        logger.warning(
                            f"Image '{image_name}' exceeds size limit "
                            f"({image_size_mb:.2f} MB > {self.options.max_image_size_mb} MB)"
                        )
                        return None
                    
                    # Convert to PNG or keep original format
                    buffer = io.BytesIO()
                    if self.options.convert_to_png:
                        pil_image.save(buffer, format='PNG', optimize=True)
                        mime_type = "image/png"
                    else:
                        # Try to preserve original format
                        original_format = pil_image.format or 'PNG'
                        pil_image.save(buffer, format=original_format, quality=self.options.compression_quality)
                        mime_type = f"image/{original_format.lower()}"
                    
                    image_data = buffer.getvalue()
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    data_uri = f"data:{mime_type};base64,{base64_data}"
                    
                    # Cache the result
                    result = (data_uri, mime_type)
                    self._image_cache[cache_key] = result
                    
                    return result
                    
                except Exception as conversion_error:
                    logger.warning(f"Failed to convert image '{image_name}': {conversion_error}")
                    return None
                
            except Exception as e:
                logger.warning(f"Failed to extract image '{image_name}': {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error hydrating image data for '{image_name}' on page {page_index + 1}: {e}")
            return None
    
    def hydrate_images_in_stream(
        self, 
        stream_elements: List[Union[ProcessingTextRun, PdfImageElement]], 
        page_index: int
    ) -> List[Union[ProcessingTextRun, PdfImageElement]]:
        """
        Hydrate image placeholders with actual data using pikepdf.
        
        Args:
            stream_elements: List of stream elements with image placeholders
            page_index: 0-based page index
        
        Returns:
            List with hydrated images (or original placeholders if hydration fails)
        """
        if not self.options.include_image_data:
            return stream_elements
        
        hydrated_elements = []
        image_count = 0
        
        for element in stream_elements:
            if isinstance(element, PdfImageElement) and element.data is None:
                # This is a placeholder - try to hydrate it
                image_data_result = self.hydrate_image_data(page_index, element.name)
                
                if image_data_result:
                    data_uri, mime_type = image_data_result
                    # Create hydrated copy
                    hydrated_image = PdfImageElement(
                        id=element.id,
                        type=element.type,
                        name=element.name,
                        imageIndex=element.imageIndex,
                        x=element.x,
                        y=element.y,
                        width=element.width,
                        height=element.height,
                        mimeType=mime_type,
                        data=data_uri
                    )
                    hydrated_elements.append(hydrated_image)
                    image_count += 1
                else:
                    # Keep placeholder if hydration fails
                    hydrated_elements.append(element)
            else:
                # Not an image placeholder, keep as-is
                hydrated_elements.append(element)
        
        if image_count > 0:
            logger.debug(f"Page {page_index + 1}: Hydrated {image_count} image(s)")
        
        return hydrated_elements
    
    def clear_cache(self) -> None:
        """Clear the image cache"""
        self._image_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_images': len(self._image_cache),
            'cache_size_bytes': sum(len(data[0]) for data in self._image_cache.values())
        }