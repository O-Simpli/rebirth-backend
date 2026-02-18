# PDFEngine - Core PDF Processing Architecture

The PDFEngine provides a unified, modular architecture for PDF processing operations. It coordinates multiple specialized processors to handle text extraction, image processing, and content modification.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        PDFEngine                             │
│  - Resource Management (PDF documents, file handles)        │
│  - Processor Coordination                                    │
│  - Configuration Management                                  │
│  - Lifecycle Management (context manager)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│TextProcessor │ │ImageProcessor│ │ ContentModifier  │
│              │ │              │ │                  │
│ - Text       │ │ - Image      │ │ - Content        │
│   Extraction │ │   Extraction │ │   Removal        │
│ - Stream     │ │ - Hydration  │ │ - Reference      │
│   Order      │ │ - Caching    │ │   Counting       │
│ - Rotated    │ │ - Format     │ │ - PDF            │
│   Text       │ │   Conversion │ │   Generation     │
│ - Grouping   │ │              │ │                  │
└──────────────┘ └──────────────┘ └──────────────────┘
```

## Core Components

### PDFEngine

The central coordinator that manages the entire PDF processing lifecycle.

**Key Responsibilities:**
- Opens and manages PDF documents
- Initializes and coordinates processors
- Provides unified API for all operations
- Manages resource cleanup

**Usage:**
```python
from engine import PDFEngine, EngineConfig

# Basic usage
with PDFEngine("document.pdf") as engine:
    # Extract text
    pages = engine.text_processor.extract_text()
    
    # Extract images
    images = engine.image_processor.extract_images()
    
    # Remove content
    modified_pdf = engine.content_modifier.remove_content(removals)

# With configuration
config = EngineConfig(
    text_processor_options=TextProcessorOptions(
        group_by_section=True,
        merge_rotated_text=True
    ),
    image_processor_options=ImageProcessorOptions(
        include_image_data=True,
        enable_caching=True
    )
)

with PDFEngine("document.pdf", config=config) as engine:
    # Use configured processors
    pages = engine.text_processor.extract_text()
```

### BaseProcessor

Abstract base class that all processors inherit from.

**Features:**
- Automatic engine reference injection
- Consistent initialization pattern
- Cleanup hooks

**Creating a New Processor:**
```python
from engine.base_processor import BaseProcessor

class MyProcessor(BaseProcessor):
    def __init__(self, engine: 'PDFEngine', options: MyProcessorOptions):
        super().__init__(engine)
        self.options = options or MyProcessorOptions()
    
    def my_operation(self):
        # Access engine resources
        doc = self.engine.pdf_document
        # Perform operation
        return result
    
    def cleanup(self):
        # Optional cleanup
        pass
```

## Processors

### TextProcessor

Extracts text from PDFs with support for horizontal and rotated text.

**Location:** `engine/text_processor.py`  
**Lines:** 448

**Key Features:**
- **Stream Order Extraction**: Preserves Z-index (paint order)
- **Rotated Text Handling**: Detects and normalizes rotated text
- **Text Grouping**: Semantic grouping with configurable options
- **Multiple Strategies**: Reading order vs. stream order

**API:**
```python
# Extract text in reading order
pages = text_processor.extract_text(start_page=1, end_page=5)

# Extract text in stream order (Z-index preserved)
pages = text_processor.extract_text_stream_order()

# Access horizontal runs for a page
runs = text_processor.get_horizontal_runs(page_index=0)

# Get rotated overlays
rotated = text_processor.get_rotated_overlays(page_index=0)

# Group content with custom config
grouped = text_processor.group_content(runs, text_config, page_num)
```

**Configuration:**
```python
TextProcessorOptions(
    group_by_section=True,      # Group text into sections
    merge_rotated_text=True,    # Merge rotated text runs
    interaction_margin=5.0,     # Pixel margin for grouping
    max_line_spacing=1.5        # Max spacing ratio for lines
)
```

**Extracted Data Structure:**
```python
{
    "type": "text",
    "text": "Content",
    "x": 100.5,
    "y": 200.3,
    "width": 150.2,
    "height": 12.0,
    "fontFamily": "Arial",
    "fontSize": 12,
    "fontWeight": 400,
    "fontStyle": "normal",
    "color": "#000000",
    "rotation": 0.0
}
```

### ImageProcessor

Extracts images from PDFs with comprehensive metadata and transformation tracking.

**Location:** `engine/image_processor.py`  
**Lines:** 280

**Key Features:**
- **Image Hydration**: Decodes and decompresses images
- **Format Conversion**: Handles multiple PDF image formats
- **Caching**: Optional image data caching for performance
- **Base64 Encoding**: For web-friendly image embedding
- **CTM Tracking**: Accurate transformation matrices

**API:**
```python
# Extract all images from document
images = image_processor.extract_images(include_data=True)

# Extract images from specific pages
images = image_processor.extract_images(
    start_page=1,
    end_page=10,
    include_data=False  # Just metadata
)

# Get single image
image = image_processor.get_image(page_num=1, image_index=0)
```

**Configuration:**
```python
ImageProcessorOptions(
    include_image_data=True,    # Include base64 image data
    enable_caching=True,        # Cache extracted images
    target_format='PNG',        # Convert to specific format
    max_image_size=10000        # Max dimension limit
)
```

**Extracted Data Structure:**
```python
{
    "type": "image",
    "image_name": "Im1",
    "x": 50.0,
    "y": 100.0,
    "width": 200.0,
    "height": 150.0,
    "image_data": "data:image/png;base64,...",  # If include_data=True
    "transform": {
        "a": 1.0, "b": 0.0, "c": 0.0,
        "d": 1.0, "e": 0.0, "f": 0.0
    }
}
```

### ContentModifier

Removes text and images from PDFs while maintaining document integrity.

**Location:** `engine/content_modifier.py`  
**Lines:** 560

**Key Features:**
- **Precise Removal**: Bbox-based text removal, name-based image removal
- **Reference Counting**: Tracks resource usage across pages
- **IoU Matching**: Intersection over Union for bbox matching
- **Multi-Page Support**: Batch processing multiple pages
- **Resource Cleanup**: Removes unused resources

**API:**
```python
# Remove content from a single page
modifier.remove_content_from_page(
    page_num=1,
    text_bboxes=[{"x": 100, "y": 200, "width": 150, "height": 20}],
    image_bboxes=[{"x": 50, "y": 300, "width": 200, "height": 150}]
)

# Remove content from multiple pages
page_removals = {
    1: RemovalRequest(text_bboxes=[...], image_bboxes=[...]),
    3: RemovalRequest(text_bboxes=[...], image_bboxes=[...])
}
modifier.remove_content_from_pages(page_removals)

# Get modified PDF
pdf_bytes = modifier.get_modified_pdf_bytes()
```

**Configuration:**
```python
ContentModifierOptions(
    reference_counting=True,     # Track resource references
    overlap_threshold=0.8,       # IoU threshold for matching
    text_tolerance=2.0,          # Pixel tolerance for text
    preserve_empty_content=False # Keep empty content streams
)
```

## Configuration System

### EngineConfig

Central configuration object for the entire engine.

```python
@dataclass
class EngineConfig:
    """Configuration for PDFEngine and all processors"""
    
    # Processor options
    text_processor_options: Optional[TextProcessorOptions] = None
    image_processor_options: Optional[ImageProcessorOptions] = None
    content_modifier_options: Optional[ContentModifierOptions] = None
    
    # Global settings
    max_file_size_mb: int = 50
    enable_validation: bool = True
    log_level: str = "INFO"
```

**Usage:**
```python
# Create custom configuration
config = EngineConfig(
    text_processor_options=TextProcessorOptions(
        group_by_section=True
    ),
    image_processor_options=ImageProcessorOptions(
        include_image_data=False,
        enable_caching=True
    ),
    content_modifier_options=ContentModifierOptions(
        reference_counting=True
    ),
    max_file_size_mb=100
)

# Use configuration
with PDFEngine("doc.pdf", config=config) as engine:
    # Processors use configured options
    pass
```

## Resource Management

The engine uses Python context managers for automatic resource cleanup:

```python
# Automatic cleanup
with PDFEngine("document.pdf") as engine:
    data = engine.text_processor.extract_text()
    # PDF is closed automatically when context exits

# Manual cleanup
engine = PDFEngine("document.pdf")
try:
    data = engine.text_processor.extract_text()
finally:
    engine.close()  # Explicit cleanup
```

**What Gets Cleaned Up:**
- PDF document handles (pdfplumber, pikepdf)
- File handles
- Processor internal state
- Cached data
- Temporary resources

## Error Handling

All operations raise appropriate exceptions:

```python
from utils.validation import PdfValidationError

try:
    with PDFEngine("document.pdf") as engine:
        pages = engine.text_processor.extract_text()
except FileNotFoundError:
    # PDF file not found
    pass
except PdfValidationError as e:
    # PDF validation or processing error
    print(f"PDF error: {e}")
except Exception as e:
    # Unexpected error
    print(f"Error: {e}")
```

## Performance Considerations

### Memory Management
- **Streaming**: Processes pages individually to reduce memory
- **Caching**: Optional image caching with configurable limits
- **Cleanup**: Aggressive resource cleanup via context managers

### Optimization Tips
```python
# For large PDFs, process in batches
with PDFEngine("large.pdf") as engine:
    total_pages = engine.get_page_count()
    batch_size = 10
    
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        pages = engine.text_processor.extract_text(
            start_page=start,
            end_page=end
        )
        # Process batch
        process_pages(pages)

# Disable image data for faster extraction
config = EngineConfig(
    image_processor_options=ImageProcessorOptions(
        include_image_data=False  # Just metadata
    )
)
```

## Testing

Each processor has comprehensive test coverage:

- **Phase 1 Tests**: Core engine (5/5 passing)
- **Phase 2 Tests**: TextProcessor (5/5 passing)  
- **Phase 3 Tests**: ImageProcessor (5/5 passing)
- **Phase 4 Tests**: ContentModifier (5/5 passing)
- **Phase 5 Tests**: API integration (6/6 passing)

**Run Tests:**
```bash
# Test specific phase
python3 scripts/test_phase_1.py  # Core engine
python3 scripts/test_phase_2.py  # Text processor
python3 scripts/test_phase_3.py  # Image processor
python3 scripts/test_phase_4.py  # Content modifier

# Test API endpoints
python3 scripts/test_phase_5_api.py
```

## Extending the Engine

### Adding a New Processor

1. **Create processor class:**
```python
from engine.base_processor import BaseProcessor

class MyProcessor(BaseProcessor):
    def __init__(self, engine, options=None):
        super().__init__(engine)
        self.options = options or MyProcessorOptions()
```

2. **Add to engine initialization:**
```python
# In engine/pdf_engine.py
def _initialize_processors(self):
    # ... existing processors ...
    if self.config.my_processor_options:
        self.my_processor = MyProcessor(
            self,
            self.config.my_processor_options
        )
```

3. **Add configuration:**
```python
# In engine/config.py
@dataclass
class MyProcessorOptions(ProcessorOptions):
    option1: bool = True
    option2: int = 10
```

### Best Practices

1. **Use Context Managers**: Always use `with` statements
2. **Configure Once**: Create config before engine instantiation
3. **Handle Errors**: Use try/except for PDF operations
4. **Clean Resources**: Rely on automatic cleanup
5. **Test Thoroughly**: Write tests for new processors

## Migration from Legacy Code

The old `utils/pdf_engine.py` (1500+ lines) has been completely replaced with this modular architecture. See `ARCHITECTURE_REFACTOR_PLAN.md` for migration details.

**Key Improvements:**
- 4x smaller codebase per module (350-560 lines vs 1500)
- Clear separation of concerns
- Independent processor testing
- Composition over inheritance
- Easy to extend and maintain

## See Also

- [ARCHITECTURE_REFACTOR_PLAN.md](../ARCHITECTURE_REFACTOR_PLAN.md) - Complete refactor documentation
- [PHASE_5_COMPLETION.md](../PHASE_5_COMPLETION.md) - API migration details
- [Phase Test Scripts](../scripts/) - Test suites for each phase
