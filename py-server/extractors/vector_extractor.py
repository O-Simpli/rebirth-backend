"""
Vector Extractor

Public-facing API for PDF vector graphics extraction.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

from engine import PDFEngine, EngineConfig
from engine.config import VectorProcessorOptions
from models.pdf_types import PdfVectorElement

if TYPE_CHECKING:
    from models.pdf_types import PdfVectorExtractionOptions

logger = logging.getLogger(__name__)


def extract_vectors(
    file_path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    options: Optional[VectorProcessorOptions] = None,
    vector_config: Optional['PdfVectorExtractionOptions'] = None,
) -> List[List[PdfVectorElement]]:
    """Extract vector graphics from PDF file."""
    try:
        logger.info(
            f"Starting vector extraction from {file_path} "
            f"(pages {start_page} to {end_page or 'end'})"
        )
        
        # Convert simplified PdfVectorExtractionOptions to full VectorProcessorOptions
        processor_options = options
        if vector_config is not None:
            # Map simplified API config to detailed engine options with smart defaults
            processor_options = VectorProcessorOptions(
                # Always enabled for accurate extraction
                extract_gradients=True,
                extract_form_xobjects=True,
                preserve_color_spaces=True,
                generate_svg=True,
                generate_path_data=True,
                
                # User-configurable options
                simplify_paths=vector_config.simplify_paths,
                include_pattern_image_data=vector_config.include_pattern_image_data,
                split_sparse_vectors=vector_config.split_sparse_vectors,
                enable_vector_grouping=vector_config.enable_vector_grouping,
                
                # Smart defaults for sparse vector splitting
                sparse_vector_coverage_threshold=20.0,
                sparse_vector_merge_distance=10.0,
                
                # Overlap and spatial grouping
                enable_overlap_grouping=False,
                overlap_check_window=0,
                overlap_method="area",
                overlap_threshold=0.0,
                proximity_threshold=None,
                enable_post_processing_merge=False
            )
        
        # Create engine config with only vector processor enabled
        config = EngineConfig(
            enable_text_processor=False,
            enable_image_processor=False,
            enable_content_modifier=False,
            enable_vector_processor=True,
            vector_processor_options=processor_options.to_dict() if processor_options else None
        )
        
        # Use PDFEngine for proper resource management
        with PDFEngine(file_path, config=config) as engine:
            # Extract vectors using integrated processor
            vectors = engine.vector_processor.extract_vectors(
                start_page=start_page,
                end_page=end_page,
            )
            
            logger.info(f"Extracted vectors from {len(vectors)} pages")
            
            return vectors
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Vector extraction failed: {e}", exc_info=True)
        raise