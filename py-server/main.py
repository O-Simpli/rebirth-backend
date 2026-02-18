"""PDF Content Extractor Python Server"""

import sys
import logging
import asyncio
import hashlib
import json
import base64
from typing import Optional, List, Dict, Set
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from rich.console import Console
from rich.logging import RichHandler

from models.pdf_types import (
    PdfElement, 
    RemovalRequest, 
    ProcessPdfConfig, 
    ProcessPdfResponse, 
    PdfImageElement,
    PdfVectorElement,
    PdfVectorExtractionOptions
)
from extractors.image_extractor import detect_image_mime_type, extract_images, get_image
from extractors.pdf_modifier import remove_content_from_pdf
from extractors.text_extractor import extract_text, extract_text_with_stream_order
from extractors.vector_extractor import extract_vectors
from utils.endpoint_decorators import handle_pdf_processing

API_VERSION = "1.0.0"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://studiosimpli.com",
]
DEFAULT_CACHE_MAX_AGE = 3600
DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 600

logger = logging.getLogger("rich")

app = FastAPI(
    title="PDF Content Extractor API",
    description="Extract structured content from PDF files",
    version=API_VERSION
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PDF Content Extractor API", 
        "version": API_VERSION,
        "features": [
            "PDF text extraction (reading-order and stream-order)",
            "PDF image extraction",
            "PDF vector extraction (paths, gradients, shadings)",
            "Content removal (text, images, vectors)",
            "Multiple bit depth support",
            "Multiple color space support",
            "Alpha channel and transparency support"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check with dependency verification"""
    try:
        import PIL
        import pdfminer
        import pikepdf
        import numpy
        
        return {
            "status": "healthy",
            "version": API_VERSION,
            "features": {
                "text_extraction": "pdfminer.six",
                "image_extraction": "pdfminer.six",
                "vector_extraction": "pikepdf",
                "pdf_manipulation": "pikepdf"
            },
            "dependencies": {
                "PIL": PIL.__version__,
                "pdfminer": pdfminer.__version__,
                "pikepdf": pikepdf.__version__,
                "numpy": numpy.__version__
            }
        }
    except ImportError as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": f"Missing dependency: {str(e)}"
            }
        )

@app.post("/extract-pdf-text", response_model=List[List[PdfElement]])
@handle_pdf_processing
async def extract_pdf_text(
    *,
    request: Request,
    file: UploadFile = File(...),
    enable_grouping: Optional[bool] = Form(True, description="Enable text grouping algorithms"),
    enable_list_detection: Optional[bool] = Form(True, description="Enable list detection"),
    enable_toc_detection: Optional[bool] = Form(True, description="Enable table of contents detection"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Extract structured text from PDF with optional grouping.
    
    **Extraction Method:**
    - Uses reading-order extraction (top-to-bottom, left-to-right)
    - For stream-order with Z-index preservation, use [/process-pdf](#/default/process_pdf_process_pdf_post)
    
    **Configuration Options:**
    - `enable_grouping`: Group text elements into semantic blocks (default: `true`)
    - `enable_list_detection`: Detect bulleted/numbered lists (default: `true`)
    - `enable_toc_detection`: Detect table of contents structure (default: `true`)
    
    **Returns:**
    - Array of element lists (one per page) with position, font, and styling information
    """
    temp_file_path = request.state.temp_file_path
    
    logger.info(f"Extracting text from PDF with grouping={enable_grouping}")
    
    result = await asyncio.to_thread(
        extract_text,
        temp_file_path,
        start_page=1,
        end_page=None,
        text_config={
            'enable_grouping': enable_grouping,
            'enable_list_detection': enable_list_detection,
            'enable_toc_detection': enable_toc_detection,
        }
    )
    
    if result is None:
        logger.error(f"PDF extraction returned None")
        raise HTTPException(
            status_code=500,
            detail="PDF extraction failed - no content could be extracted"
        )
    
    logger.info(f"Successfully extracted text from {len(result)} pages")
    return result

@app.post("/extract-pdf-images", response_model=List[List[PdfElement]])
@handle_pdf_processing
async def extract_pdf_images(
    *,
    request: Request,
    file: UploadFile = File(...),
    include_image_data: Optional[bool] = Form(False, description="Include base64-encoded image data in response"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Extract image metadata from PDF.
    
    **Configuration:**
    - `include_image_data=false` (default): Returns metadata only (position, dimensions, format)
    - `include_image_data=true`: Includes base64-encoded image data
    
    **Returns:**
    - Array of element lists (one per page)
    - Each element contains: position, dimensions, format, ID, and optionally base64 data
    
    **Note:** Use [/extract-image](#/default/extract_image_extract_image_post) endpoint to fetch individual images by ID.
    """
    temp_file_path = request.state.temp_file_path
    
    logger.info(f"Extracting images from PDF (include_data={include_image_data})")
    
    all_pages_data = await asyncio.to_thread(
        extract_images,
        temp_file_path,
        start_page=1,
        end_page=None,
        include_image_data=include_image_data
    )

    logger.info(f"Successfully extracted image metadata from {len(all_pages_data)} pages")
    return all_pages_data

@app.post("/extract-pdf-vectors", response_model=List[List[PdfVectorElement]])
@handle_pdf_processing
async def extract_pdf_vectors(
    *,
    request: Request,
    file: UploadFile = File(...),
    start_page: Optional[int] = Query(1, ge=1, description="Starting page number (1-based)"),
    end_page: Optional[int] = Query(None, ge=1, description="Ending page number (1-based), None for all pages"),
    config: Optional[str] = Form(None, description="Optional JSON string containing PdfVectorExtractionOptions configuration"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Extract vector graphics from PDF with gradient and pattern support.
    
    **Returns:**
    - Array of vector element lists (one per page) with:
      - SVG content with embedded gradients
      - Path data for direct rendering
      - Position, dimensions, opacity
      - Stroke and fill information
    
    **Supported Features:**
    - Linear and radial gradients (ShadingType 2, 3)
    - Pattern fills (PatternType 2)
    - Complex color spaces (DeviceRGB, CMYK, Gray, ICCBased, Indexed)
    - Form XObjects (opacity-affected vectors)
    - Clipping paths
    
    **Configuration (JSON):**
    - `include_pattern_image_data`: Include base64 data for pattern images (default: `false`)
    - `simplify_paths`: Simplify path data (default: `false`)
    - `split_sparse_vectors`: Split sparse vector groups (default: `true`)
    - `enable_vector_grouping`: Group overlapping/nearby vectors (default: `true`)
    """
    temp_file_path = request.state.temp_file_path
    
    # Parse vector config if provided
    vector_config = None
    if config:
        try:
            config_dict = json.loads(config)
            vector_config = PdfVectorExtractionOptions(**config_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in config: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config structure: {str(e)}")
    
    logger.info(f"Extracting vectors from PDF (pages {start_page} to {end_page or 'end'})")
    
    all_pages_vectors = await asyncio.to_thread(
        extract_vectors,
        temp_file_path,
        start_page=start_page,
        end_page=end_page,
        vector_config=vector_config
    )
    
    total_vectors = sum(len(page) for page in all_pages_vectors)
    logger.info(f"Successfully extracted {total_vectors} vectors from {len(all_pages_vectors)} pages")
    
    return all_pages_vectors

@app.post("/extract-image")
@handle_pdf_processing
async def extract_image(
    *,
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Query(..., ge=1, description="Page number (1-based)"),
    image_id: str = Query(..., description="ID of the image to extract"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Extract specific image from PDF page.
    
    **Parameters:**
    - `page_number`: Page number (1-based)
    - `image_id`: Image identifier (e.g., `"Im1"`, `"Im2"`)
    
    **Returns:**
    - Binary image data with appropriate Content-Type
    - Response headers: `ETag`, `Cache-Control` (1 hour)
    
    **Use Case:** Fetch individual images after using `/extract-pdf-images` to get metadata.
    """
    temp_file_path = request.state.temp_file_path
    file_content = request.state.file_content
    
    content_hash = hashlib.md5(file_content).hexdigest()[:16]
    etag = f'"{content_hash}-p{page_number}-id{image_id[:8]}"'
    
    logger.info(f"Extracting image {image_id} from page {page_number}")
    
    image_bytes = await asyncio.to_thread(
        get_image,
        temp_file_path,
        page_number,
        image_id
    )
    
    if not image_bytes:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found on page {page_number}"
        )
    
    mime_type = detect_image_mime_type(image_bytes)
    
    logger.info(f"Successfully extracted image {image_id} from page {page_number} ({len(image_bytes)} bytes, {mime_type})")
    
    return Response(
        content=image_bytes,
        media_type=mime_type,
        headers={
            "ETag": etag,
            "Cache-Control": f"public, max-age={DEFAULT_CACHE_MAX_AGE}",
            "Content-Length": str(len(image_bytes))
        }
    )

@app.post("/remove-pdf-content")
@handle_pdf_processing
async def remove_pdf_content(
    *,
    request: Request,
    file: UploadFile = File(...),
    removal_data: str = Form(..., description="JSON string containing page-keyed removal request data"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Remove text, images, and vectors from PDF pages.
    
    Args:
        removal_data: JSON string with page-keyed removal requests.
            Format: {"page_num": {"text_bboxes": [...], "image_bboxes": [...], "vector_bboxes": [...]}}
    
    Returns modified PDF with specified content removed.
    """
    try:
        removal_data_dict = json.loads(removal_data)
        if not isinstance(removal_data_dict, dict):
            raise ValueError("removal_data must be a dictionary")
        
        page_removals: Dict[int, RemovalRequest] = {}
        for page_str, removal_obj in removal_data_dict.items():
            try:
                page_num = int(page_str)
                if page_num < 1:
                    raise ValueError(f"Page number must be >= 1, got {page_num}")
                page_removals[page_num] = RemovalRequest(**removal_obj)
            except ValueError as e:
                raise ValueError(f"Invalid page number '{page_str}': {e}")
                
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in removal_data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid removal request data: {str(e)}")
    
    temp_file_path = request.state.temp_file_path
    
    logger.info(f"Removing content from PDF ({len(page_removals)} pages to modify)")
    
    for page_num, removal_req in page_removals.items():
        if removal_req.text_bboxes:
            logger.debug(f"Page {page_num}: Text to remove: {len(removal_req.text_bboxes)}")
        if removal_req.image_bboxes:
            logger.debug(f"Page {page_num}: Images to remove: {len(removal_req.image_bboxes)}")
        if removal_req.vector_bboxes:
            logger.debug(f"Page {page_num}: Vectors to remove: {len(removal_req.vector_bboxes)}")
    
    modified_pdf_bytes = await asyncio.to_thread(
        remove_content_from_pdf,
        temp_file_path,
        page_removals
    )
    
    if not modified_pdf_bytes:
        raise HTTPException(status_code=500, detail="Failed to remove content from PDF.")
    
    logger.info(f"Successfully removed content from PDF")
    
    filename = file.filename if file.filename else "document.pdf"
    return Response(
        content=modified_pdf_bytes,
        media_type='application/pdf',
        headers={
            "Content-Disposition": f"attachment; filename=modified_{filename}"
        }
    )

@app.post("/process-pdf", response_model=ProcessPdfResponse)
@handle_pdf_processing
async def process_pdf(
    *,
    request: Request,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None, description="Optional JSON string containing configuration for text, image, and vector processing"),
    processing_timeout: Optional[int] = Query(DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS, description="Processing timeout in seconds")
):
    """
    Extract text, images, and vectors in stream order. Generate cleaned PDF.
    
    **Extraction Method:**
    - Uses stream-order extraction with Z-index preservation
    - For reading order (top-to-bottom, left-to-right), use [/extract-pdf-text](#/default/extract_pdf_text_extract_pdf_text_post)
    
    **Configuration (JSON):**
    - `extract_text`: Enable/disable text extraction (default: `true`)
    - `extract_images`: Enable/disable image extraction (default: `true`)
    - `extract_vectors`: Enable/disable vector extraction (default: `true`)
    - `text_config`: Configuration for text extraction (see [/extract-pdf-text](#/default/extract_pdf_text_extract_pdf_text_post))
    - `image_config`: Configuration for image extraction (see [/extract-pdf-images](#/default/extract_pdf_images_extract_pdf_images_post))
    - `vector_config`: Configuration for vector extraction (see [/extract-pdf-vectors](#/default/extract_pdf_vectors_extract_pdf_vectors_post))
    
    **Returns:**
    - `processedPages`: Stream-ordered text and image elements
    - `vectors`: Vector graphics (if `extract_vectors=true`)
    - `modifiedPdf`: Base64-encoded cleaned PDF with all content removed
    
    **Performance Tip:** Disable unused extraction types to improve processing speed.
    """
    temp_file_path = request.state.temp_file_path
    
    if config:
        try:
            config_dict = json.loads(config)
            process_config = ProcessPdfConfig(**config_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in config: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config structure: {str(e)}")
    else:
        process_config = ProcessPdfConfig()
    
    # Prepare extraction tasks based on config
    extraction_tasks = []
    
    # Count enabled extraction types
    enabled_count = sum([
        process_config.extract_text,
        process_config.extract_images,
        process_config.extract_vectors
    ])
    
    # Only use stream order extraction if:
    # - At least 2 extraction types are enabled (need stream order for merging), OR
    # - Only text is enabled (text benefits from stream order)
    use_stream_order = enabled_count >= 2 or (enabled_count == 1 and process_config.extract_text)
    
    if use_stream_order:
        # Pass extraction flags to text extractor so it skips creating placeholders
        text_config = process_config.text_config.model_dump() if process_config.extract_text else {}
        text_config['skip_text_placeholders'] = not process_config.extract_text
        text_config['skip_image_placeholders'] = not process_config.extract_images
        text_config['skip_vector_placeholders'] = not process_config.extract_vectors
        
        extraction_tasks.append((
            'text',
            asyncio.to_thread(
                extract_text_with_stream_order,
                temp_file_path,
                start_page=1,
                end_page=None,
                text_config=text_config
            )
        ))
    
    if process_config.extract_images:
        extraction_tasks.append((
            'images',
            asyncio.to_thread(
                extract_images,
                temp_file_path,
                start_page=1,
                end_page=None,
                include_image_data=process_config.image_config.include_image_data
            )
        ))
    
    if process_config.extract_vectors:
        extraction_tasks.append((
            'vectors',
            asyncio.to_thread(
                extract_vectors,
                temp_file_path,
                start_page=1,
                end_page=None,
                vector_config=process_config.vector_config
            )
        ))
    
    # Log what we're extracting
    enabled_extractions = [name for name, _ in extraction_tasks]
    logger.info(f"Starting parallel extraction: {', '.join(enabled_extractions)}")
    
    # Execute all extraction tasks in parallel
    results = await asyncio.gather(*[task for _, task in extraction_tasks])
    
    # Map results back to their types
    result_dict = dict(zip([name for name, _ in extraction_tasks], results))
    
    stream_pages = result_dict.get('text', [])
    rich_image_pages = result_dict.get('images', [])
    vector_pages = result_dict.get('vectors', None)
    
    # Calculate totals for logging
    total_stream_elements = sum(len(page) for page in stream_pages) if stream_pages else 0
    total_rich_images = sum(len(page) for page in rich_image_pages) if rich_image_pages else 0
    total_vectors = sum(len(page) for page in vector_pages) if vector_pages else 0
    
    # Log extraction summary
    extraction_summary = []
    if stream_pages:
        extraction_summary.append(f"{total_stream_elements} stream elements")
    if rich_image_pages:
        extraction_summary.append(f"{total_rich_images} rich images")
    if vector_pages:
        extraction_summary.append(f"{total_vectors} vectors")
    
    num_pages = max(len(stream_pages) if stream_pages else 0,
                    len(rich_image_pages) if rich_image_pages else 0,
                    len(vector_pages) if vector_pages else 0)
    
    logger.info(f"Extracted {', '.join(extraction_summary)} from {num_pages} pages")

    # Build lookup map: page_num -> image_name -> PdfImageElement
    rich_image_map: Dict[int, Dict[str, PdfImageElement]] = {}
    for page_idx, page_images in enumerate(rich_image_pages):
        page_num = page_idx + 1
        rich_image_map[page_num] = {}
        for img in page_images:
            # Normalize image name (remove leading slash if present)
            img_name = img.name.lstrip('/')
            rich_image_map[page_num][img_name] = img
    
    # Build vector lookup map: page_num -> list of vectors
    rich_vector_map: Dict[int, List[PdfVectorElement]] = {}
    if vector_pages:
        for page_idx, page_vectors in enumerate(vector_pages):
            page_num = page_idx + 1
            rich_vector_map[page_num] = page_vectors
    
    # Merge: Replace placeholders with rich data
    final_pages: List[List[PdfElement]] = []
    total_merged_images = 0
    total_merged_vectors = 0
    total_orphan_images = 0
    total_orphan_vectors = 0
    
    # Determine number of pages from whichever extraction was enabled
    num_pages = max(
        len(stream_pages) if stream_pages else 0,
        len(rich_image_pages) if rich_image_pages else 0,
        len(vector_pages) if vector_pages else 0
    )
    
    for page_idx in range(num_pages):
        page_num = page_idx + 1
        stream_elements = stream_pages[page_idx] if stream_pages and page_idx < len(stream_pages) else []
        page_rich_images = rich_image_map.get(page_num, {})
        page_rich_vectors = rich_vector_map.get(page_num, [])
        merged_elements: List[PdfElement] = []
        used_image_names: Set[str] = set()
        used_vector_indices: Set[int] = set()
        
        for element in stream_elements:
            if element.type == 'image':
                # This is an image placeholder from the stream
                placeholder_name = element.name.lstrip('/')
                
                if placeholder_name in page_rich_images:
                    # Replace placeholder with rich image data
                    rich_image = page_rich_images[placeholder_name]
                    merged_elements.append(rich_image)
                    used_image_names.add(placeholder_name)
                    total_merged_images += 1
                # Skip placeholder if no rich data found (don't send placeholders to client)
            elif element.type == 'vector':
                # This is a vector placeholder - match by proximity
                best_match_idx = None
                best_match_score = float('inf')
                
                for vec_idx, rich_vector in enumerate(page_rich_vectors):
                    if vec_idx in used_vector_indices:
                        continue
                    
                    # Calculate distance (position + size difference)
                    pos_dist = abs(element.x - rich_vector.x) + abs(element.y - rich_vector.y)
                    size_dist = abs(element.width - rich_vector.width) + abs(element.height - rich_vector.height)
                    total_dist = pos_dist + size_dist
                    
                    # Match threshold: within 5 units total difference
                    if total_dist < 5.0 and total_dist < best_match_score:
                        best_match_score = total_dist
                        best_match_idx = vec_idx
                
                if best_match_idx is not None:
                    # Replace placeholder with rich vector data
                    rich_vector = page_rich_vectors[best_match_idx]
                    merged_elements.append(rich_vector)
                    used_vector_indices.add(best_match_idx)
                    total_merged_vectors += 1
                # Skip placeholder if no match found
            else:
                # Text element - keep as is
                merged_elements.append(element)
        
        # Handle orphans: images found by extract_images but not in stream
        for img_name, rich_image in page_rich_images.items():
            if img_name not in used_image_names:
                merged_elements.append(rich_image)
                total_orphan_images += 1
        
        # Handle orphans: vectors found by extract_vectors but not in stream
        for vec_idx, rich_vector in enumerate(page_rich_vectors):
            if vec_idx not in used_vector_indices:
                merged_elements.append(rich_vector)
                total_orphan_vectors += 1
        
        final_pages.append(merged_elements)
    
    logger.info(f"Merge complete: {total_merged_images} images enriched, {total_merged_vectors} vectors enriched")
    if total_orphan_images > 0 or total_orphan_vectors > 0:
        logger.debug(f"Orphans appended: {total_orphan_images} images, {total_orphan_vectors} vectors")
    
    page_removals: Dict[int, RemovalRequest] = {}
    for page_index, page_elements in enumerate(final_pages):
        page_num = page_index + 1
        text_bboxes = []
        image_bboxes = []
        
        for element in page_elements:
            if element.type == 'image':
                from models.pdf_types import ImageRemovalRequest, BoundingBox
                image_bboxes.append(ImageRemovalRequest(
                    name=element.name,
                    bbox=BoundingBox(
                        x=element.x,
                        y=element.y,
                        width=element.width,
                        height=element.height
                    )
                ))
            else:
                from models.pdf_types import BoundingBox
                text_bboxes.append(BoundingBox(
                    x=element.x,
                    y=element.y,
                    width=element.width,
                    height=element.height
                ))
        
        # Add vector bboxes if vectors were extracted
        vector_bboxes = []
        if vector_pages and page_index < len(vector_pages):
            from models.pdf_types import BoundingBox
            for vector in vector_pages[page_index]:
                vector_bboxes.append(BoundingBox(
                    x=vector.x,
                    y=vector.y,
                    width=vector.width,
                    height=vector.height
                ))
        
        if text_bboxes or image_bboxes or vector_bboxes:
            page_removals[page_num] = RemovalRequest(
                text_bboxes=text_bboxes if text_bboxes else None,
                image_bboxes=image_bboxes if image_bboxes else None,
                vector_bboxes=vector_bboxes if vector_bboxes else None
            )
    
    logger.info(f"Generating cleaned PDF ({len(page_removals)} pages to modify)")
    modified_pdf_bytes = await asyncio.to_thread(
        remove_content_from_pdf,
        temp_file_path,
        page_removals
    )
    
    modified_pdf_base64 = base64.b64encode(modified_pdf_bytes).decode('utf-8')
    
    logger.info(f"Process PDF complete: {len(final_pages)} pages processed")
    return ProcessPdfResponse(
        processedPages=final_pages,
        modifiedPdf=modified_pdf_base64,
        vectors=vector_pages
    )

def _configure_server_logging():
    """Configure logging with Rich handler and filters for clean output"""
    console = Console(force_terminal=True)
    
    # Get level from env, default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    class ShutdownFilter(logging.Filter):
        """Filter out shutdown-related log messages"""
        def filter(self, record):
            if record.exc_info and record.exc_info[0] in (KeyboardInterrupt, asyncio.CancelledError):
                return False
            if "CancelledError" in str(record.msg) or "KeyboardInterrupt" in str(record.msg):
                return False
            return True

    rich_handler = RichHandler(
        console=console, 
        rich_tracebacks=True,
        show_time=False,
        show_path=True
    )
    rich_handler.addFilter(ShutdownFilter())
    
    # Silence everything by default
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[rich_handler])

    # Allow server startup logs
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Set specific module log levels
    for module_name in ["main", "rich", "engine", "extractors", "processors", "utils"]:
        logging.getLogger(module_name).setLevel(log_level)

    return console

def _find_free_port(start_port: int = 8000) -> int:
    """Find an available port starting from the given port"""
    import socket
    
    port = start_port
    max_port = start_port + 100
    
    while port < max_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    
    return start_port

server_console = _configure_server_logging()

if __name__ == "__main__":
    free_port = _find_free_port()
    server_console.print(f"[bold green]🚀 Starting server on http://localhost:{free_port}[/bold green]")
    
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=free_port, reload=True, log_config=None)
    except KeyboardInterrupt:
        server_console.print("\n[bold yellow]🛑 Server stopped.[/bold yellow]")
        sys.exit(0)