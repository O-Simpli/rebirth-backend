"""
Pydantic models for PDF Content Extraction API
Mirrors the TypeScript interfaces from the original implementation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Any, Union, Literal, Dict
from enum import Enum

class TextGroupingReason(str, Enum):
    """Enumeration for text grouping reasons - matches TypeScript"""
    HORIZONTAL_PROXIMITY = "horizontal-proximity"
    VERTICAL_PROXIMITY = "vertical-proximity"
    FONT_SIMILARITY = "font-similarity"
    SEMANTIC_BLOCK = "semantic-block"
    LIST_GROUPING = "list-grouping"
    TABLE_OF_CONTENTS = "table-of-contents"
    TABLE_CELL = "table-cell"
    MANUAL_OVERRIDE = "manual-override"

class SectionType(str, Enum):
    """Enumeration for document section types - matches TypeScript"""
    PARAGRAPH = "paragraph"
    LIST = "list"
    HEADER = "header"
    TABLE_OF_CONTENTS = "table-of-contents"

# Base models for PDF elements
class BoundingBox(BaseModel):
    """Bounding box for grouped elements"""
    x: float
    y: float
    width: float
    height: float

class PdfElementBase(BoundingBox):
    """Base class for all PDF elements"""
    id: str
    rotation: float = 0.0
    stream_index: Optional[int] = None  # Track original stream position for z-order preservation

class PdfTextRun(BaseModel):
    """Individual text run with styling information"""
    text: str

    # Font styling properties
    fontFamily: Optional[str] = None
    fontWeight: Optional[str] = "normal"
    fontStyle: Optional[str] = "normal"
    fontSize: Optional[float] = None
    color: Optional[str] = None
    textDecoration: Optional[str] = "none"
    originalFontName: Optional[str] = None
    stream_index: Optional[int] = None  # Track original stream position for ordering

class ProcessingTextRun(PdfTextRun, BoundingBox):
    """
    Internal text run with full positional data for grouping calculations.
    Inherits styling properties from PdfTextRun and positional data from BoundingBox.
    This is used during the processing pipeline and is not exposed in the API.
    """
    rotation: float = 0.0
    stream_index: Optional[int] = None  # Original position in content stream for stream order preservation
    tm_start_x: Optional[float] = None  # Text matrix X position at start of render_string
    tm_end_x: Optional[float] = None    # Text matrix X position at end of render_string

class PdfImageElement(PdfElementBase):
    """Image element with positioning and optional base64 data"""
    type: Literal["image"] = "image"
    name: str  # PDF XObject name (e.g., 'Im1', 'Im2', etc.)
    imageIndex: int
    mimeType: str = "image/unknown"
    estimatedSize: Optional[str] = None
    format: Optional[str] = None  # Image format (PNG, JPEG, etc.)
    data: Optional[str] = None  # Base64 encoded image data (optional for stateless)
    skewX: float = 0.0
    skewY: float = 0.0

class PatternImage(BaseModel):
    """Image embedded within a PDF pattern"""
    name: str  # XObject name (e.g., "I1")
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    data: Optional[str] = None  # Base64 encoded image data (optional)

class PdfVectorElement(PdfElementBase):
    """Vector element with SVG content and path data for client-side rendering"""
    type: Literal["vector"] = "vector"
    svgContent: str  # Complete SVG markup for fallback rendering
    pathData: Optional[str] = None  # Extracted SVG 'd' attribute for direct Konva.Path rendering
    opacity: float = 1.0  # Combined fill/stroke opacity
    stroke: Optional[str] = None  # Stroke color
    fill: Optional[str] = None  # Fill color
    strokeWidth: Optional[float] = None  # Stroke width
    
    # Pattern fill support (Type 1 Tiling Patterns with embedded images)
    fillType: Optional[Literal["solid", "gradient", "pattern"]] = None  # Type of fill
    patternType: Optional[int] = None  # PDF PatternType (1=Tiling, 2=Shading)
    patternImages: Optional[List[PatternImage]] = None  # Images embedded in pattern (for Type 1)
    
    # Clipping region grouping - explicit logical grouping via W/W* operators
    clippingRegionId: Optional[int] = None
    clippingPath: Optional[str] = None  # SVG path 'd' attribute of the clipping boundary
    clippingRule: Optional[str] = None  # "nonzero" or "evenodd"

# Document section models
class BaseContentSection(BaseModel):
    """Base class for all document sections"""
    type: SectionType
    indentationLevel: float
    sectionId: str

class ParagraphSection(BaseContentSection):
    """Paragraph section containing lines of text runs"""
    type: Literal[SectionType.PARAGRAPH] = SectionType.PARAGRAPH
    lines: List[List[PdfTextRun]]
    fontSize: Optional[float] = None
    lineHeight: Optional[float] = None
    marginBottom: Optional[float] = None
    textAlign: Optional[Literal["left", "center", "right", "justify"]] = None

class OrderedListMetadata(BaseModel):
    """Metadata for ordered lists - matches TypeScript OrderedListMetadata"""
    type: Literal["ordered"] = "ordered"
    subtype: Literal["decimal", "letter", "roman", "letter-upper", "roman-upper"]
    startingNumber: int
    startingLetter: Optional[str] = None
    nestingLevel: int
    parentListId: Optional[str] = None

class UnorderedListMetadata(BaseModel):
    """Metadata for unordered lists - matches TypeScript UnorderedListMetadata"""
    type: Literal["unordered"] = "unordered"
    bulletCharacter: str
    nestingLevel: int
    parentListId: Optional[str] = None

class ListSection(BaseContentSection):
    """List section containing paragraphs with list metadata"""
    type: Literal[SectionType.LIST] = SectionType.LIST
    paragraphs: List[ParagraphSection]
    metadata: Union[OrderedListMetadata, UnorderedListMetadata]

class HeaderSection(BaseContentSection):
    """Header section with optional level"""
    type: Literal[SectionType.HEADER] = SectionType.HEADER
    lines: List[List[PdfTextRun]]
    level: Optional[int] = None

class TableOfContentsSection(BaseContentSection):
    """Table of contents section"""
    type: Literal[SectionType.TABLE_OF_CONTENTS] = SectionType.TABLE_OF_CONTENTS
    lines: List[List[PdfTextRun]]

# Union type for all document sections
ContentSection = Union[ParagraphSection, ListSection, HeaderSection, TableOfContentsSection]

class PdfTextElement(PdfElementBase):
    """Text element containing multiple document sections"""
    type: Literal["text"] = "text"
    text: str
    sections: List[ContentSection]

# Union type for all PDF elements
PdfElement = Union[PdfTextElement, PdfImageElement, PdfVectorElement]

# Obstacle Models for Layout Detection 
class PdfLine(BaseModel):
    """Represents a vector line obstacle in the PDF."""
    x0: float
    y0: float
    x1: float
    y1: float
    linewidth: float
    stroke: Optional[Any] = None

class PdfRect(BaseModel):
    """Represents a vector rectangle obstacle in the PDF."""
    x0: float
    y0: float
    x1: float
    y1: float
    linewidth: float
    stroke: Optional[Any] = None
    fill: Optional[Any] = None

class PdfCurve(BaseModel):
    """Represents a vector curve (e.g., Bezier) obstacle in the PDF."""
    points: List[Tuple[float, float]]
    linewidth: float
    stroke: Optional[Any] = None
    fill: Optional[Any] = None

class PdfObstacles(BaseModel):
    """Container for all detected obstacle elements on a page."""
    lines: List[PdfLine] = Field(default_factory=list)
    rects: List[PdfRect] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None

# Configuration models
class PdfTextExtractionOptions(BaseModel):
    """Simplified configuration for text extraction - high-level features only"""
    enable_grouping: bool = Field(True, description="Group text into structured paragraphs and sections")
    enable_list_detection: bool = Field(True, description="Detect and structure bulleted and numbered lists")
    enable_toc_detection: bool = Field(True, description="Detect and structure table of contents sections")

class PdfImageExtractionOptions(BaseModel):
    """Configuration for image extraction"""
    include_image_data: bool = Field(False, description="If true, embed base64 image data directly in the response")

class PdfVectorExtractionOptions(BaseModel):
    """Simplified configuration for vector graphics extraction"""
    # Payload control
    include_pattern_image_data: bool = Field(False, description="Include base64 image data for images embedded in tiling patterns")
    
    # Optimizations
    simplify_paths: bool = Field(False, description="Simplify complex paths (may reduce quality but improves performance)")
    split_sparse_vectors: bool = Field(True, description="Split large sparse vectors into content-aware bounds (eliminates empty space)")
    enable_vector_grouping: bool = Field(False, description="Group vectors by clipping region (W/W* operators) - merges vectors within the same logical group")


class ImageRemovalRequest(BaseModel):
    """Request model for removing a specific image with name and position validation"""
    name: str = Field(..., description="PDF XObject name of the image (e.g., 'Im1', 'Im2')")
    bbox: BoundingBox = Field(..., description="Bounding box defining the image's location and dimensions")

class RemovalRequest(BaseModel):
    """Request model for PDF content removal - supports targeted text, image, and vector removal"""
    text_bboxes: Optional[List[BoundingBox]] = Field(None, description="List of text bounding boxes to remove")
    image_bboxes: Optional[List[ImageRemovalRequest]] = Field(None, description="List of images to remove with name and position validation")
    vector_bboxes: Optional[List[BoundingBox]] = Field(None, description="List of vector bounding boxes to remove")

class ProcessPdfConfig(BaseModel):
    """Configuration for the unified process-pdf endpoint"""
    # Top-level toggles for performance (skip entire extraction types)
    extract_text: bool = Field(True, description="Whether to extract text elements")
    extract_images: bool = Field(True, description="Whether to extract image elements")
    extract_vectors: bool = Field(True, description="Whether to extract vector graphics")
    
    # Granular configuration for each extraction type
    text_config: PdfTextExtractionOptions = Field(default_factory=PdfTextExtractionOptions, description="Configuration for text extraction and grouping")
    image_config: PdfImageExtractionOptions = Field(default_factory=PdfImageExtractionOptions, description="Configuration for image extraction")
    vector_config: PdfVectorExtractionOptions = Field(default_factory=PdfVectorExtractionOptions, description="Configuration for vector extraction")

class ProcessPdfResponse(BaseModel):
    """Response model for the unified process-pdf endpoint"""
    processedPages: List[List[PdfElement]] = Field(..., description="Array of page data - each inner list contains the elements for a single page")
    modifiedPdf: str = Field(..., description="Base64 encoded string of the cleaned/modified PDF")
    vectors: Optional[List[List[PdfVectorElement]]] = Field(default=None, description="Array of vector graphics - each inner list contains vectors for a single page (if extract_vectors=True)")