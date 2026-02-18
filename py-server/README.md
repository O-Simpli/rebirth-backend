# PDF Processing Backend

This directory contains the Python application designed to handle all low-level PDF processing tasks. The server is responsible for text extraction, image extraction, and PDF content modification.

## Features

-   **Structured Text Extraction**: Extracts text with precise Z-index layering (stream order), preserving the visual hierarchy. Uses advanced "re-distillation" for handling rotated text and groups characters into semantic `Sections` with computed margins and indentation.
-   **High-Fidelity Image Extraction**: Combines metadata mining with stream-based positioning to provide high-resolution images with exact page coordinates. Explicitly handles overlapping elements correctly.
-   **Web-Ready Coordinates**: Automatically normalizes all coordinates from PDF space (Bottom-Left origin) to Web space (Top-Left origin), simplifying frontend integration.
-   **Content-Aware PDF Modification**: Removes specified text or image elements from a PDF and regenerates the file with the content removed.
-   **Security & Resource Management**: Implements validation for file uploads (size, type) and manages server resources effectively.

## Architecture

The server is built on a modern, modular architecture based on **composition over inheritance**. This design provides clear separation of concerns, improved testability, and easier maintenance.

### Directory Structure

```
py-server/
├── main.py                      # FastAPI endpoints (orchestration only)
├── models/                      # Data models (Pydantic schemas)
│   └── pdf_types.py             # Type definitions for PDF elements
├── engine/                      # Core PDF processing engine
│   ├── __init__.py
│   ├── base_processor.py        # Abstract base class for processors
│   ├── config.py                # Engine configuration
│   ├── pdf_engine.py            # PDFEngine class (unified interface)
│   ├── text_processor.py        # TextProcessor (coordinates text extraction)
│   ├── image_processor.py       # ImageProcessor (coordinates image extraction)
│   └── content_modifier.py      # ContentModifier (coordinates content removal)
├── processors/                  # Stateful processing components
│   ├── __init__.py
│   ├── stream_order_device.py   # PDFMiner device for stream-order extraction
│   ├── text_grouping.py         # Semantic text grouping with spatial indexing
│   ├── text_processing_helpers.py # Text run accumulation and styling
│   ├── position_tracking.py     # Position calculation strategies
│   ├── pdf_graphics.py          # Graphics state tracking (CTM, opacity)
│   └── text_normalizer.py       # Rotated text normalization (re-distillation)
├── extractors/                  # High-level extraction APIs
│   ├── __init__.py
│   ├── text_extractor.py        # Public text extraction API
│   ├── image_extractor.py       # Public image extraction API
│   └── pdf_modifier.py          # Public content removal API
└── utils/                       # Pure utility functions
    ├── __init__.py
    ├── font_mapping.py          # Font name → CSS transformations
    ├── pdf_transforms.py        # Matrix math and coordinate transformations
    ├── validation.py            # File and content validation
    └── endpoint_decorators.py   # FastAPI decorator utilities
```

**Directory Purpose:**

- **`engine/`**: Orchestration layer - coordinates processors and manages PDF resources
- **`processors/`**: Stateful components with complex algorithms (devices, trackers, accumulators)
- **`extractors/`**: Public API layer - adapters that maintain backward compatibility
- **`utils/`**: Pure functions with no state - reusable helpers for specific tasks
- **`models/`**: Data structures and type definitions

### Core Architecture: PDFEngine + Processors

The architecture is centered around the `PDFEngine` class, which acts as a coordinator for specialized processor classes:

```
┌─────────────────────────────────────────────────────────────┐
│                        PDFEngine                            │
│  - Resource Management (PDF documents, file handles)        │
│  - ProcessorCoordination                                    │
│  - Configuration Management                                 │
│  - Lifecycle Management (context manager)                   │
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
│ - Stream     │ │ - CTM        │ │ - Reference      │
│   Order      │ │   Tracking   │ │   Counting       │
│ - Rotated    │ │ - Format     │ │ - PDF            │
│   Text       │ │   Conversion │ │   Generation     │
│ - Grouping   │ │ - Caching    │ │                  │
└──────────────┘ └──────────────┘ └──────────────────┘
```

**Key Design Principles:**

1. **Composition Over Inheritance**: Processors are composed into the engine rather than inheriting from it
2. **Dependency Injection**: Processors receive engine references, making them testable in isolation
3. **Single Responsibility**: Each processor handles one domain of PDF processing
4. **Adapter Pattern**: Public API maintains backward compatibility while using internal architecture
5. **Context Managers**: Automatic resource cleanup via Python's `with` statement

### Request Processing Pipeline

All endpoints leverage the `@handle_pdf_processing` decorator pattern, which encapsulates common functionality and eliminates boilerplate code. This decorator provides:

-   **Automatic File Validation**: Enforces file type (PDF only) and size limits before processing begins
-   **Secure Temporary File Management**: Creates temporary files with proper permissions and guarantees cleanup via context managers, even if exceptions occur
-   **Standardized Error Handling**: Catches and formats exceptions consistently across all endpoints (validation errors, processing errors, timeouts)
-   **Request Timeout Management**: Prevents long-running requests from consuming server resources indefinitely
-   **Memory-Safe File Handling**: Uses streaming and temporary storage to handle large PDFs without exhausting memory
-   **Internal State Communication**: Uses `request.state` to pass temporary file paths and content between the decorator and endpoint functions, keeping internal details out of public API documentation

### Multi-Page Processing Model

The server is designed to process **all pages** in the uploaded PDF file by default. This stateless, multi-page model enables:
-   Efficient batch processing (process 10 pages as easily as 1)
-   Simplified client logic (no need to orchestrate page-by-page requests)
-   Better resource utilization (amortized overhead across pages)

The client is responsible for creating PDFs containing only the pages it wants to process, using `pdf-lib` to extract specific pages from the cached document.

### Text Extraction

The text extraction process uses a sophisticated multi-stage pipeline that preserves document structure and reading order.

**Architecture:**
- **Engine**: `engine/text_processor.py` - Coordinates extraction workflow
- **Stream Device**: `processors/stream_order_device.py` - Extracts characters in paint order (Z-index)
- **Normalizer**: `processors/text_normalizer.py` - Handles rotated text via re-distillation
- **Grouping**: `processors/text_grouping.py` - Builds semantic hierarchy (lines → paragraphs → sections)

**Key Features:**

1.  **Stream-Order Processing**: Custom device extracts characters in the exact order they are painted, preserving Z-index layering. This ensures correct visual hierarchy (e.g., text on top of images).

2.  **Text Matrix Tracking**: The critical innovation - tracks text position using both:
    - `tm_start_x`: Starting X position from text matrix
    - `tm_end_x`: Ending X position after character advancement
    - This dual tracking enables accurate gap calculation for proper word spacing

3.  **Position Strategy**: Intelligent decision tree for position calculation:
    - Same line + no backwards movement + no large jump → use tracked position
    - Different line OR backwards OR large jump → use incoming position
    - Handles edge cases like faux-bold, re-rendered text, and position jumps

4.  **Coordinate Normalization**: All coordinates centralized and converted from PDF Cartesian space (Origin Bottom-Left) to Web Raster space (Origin Top-Left) using robust matrix decomposition for rotation, scale, and skew.

5.  **Advanced Handling & Re-Distillation**:
    *   **Ghost Detection**: Filters out microscopic invisible text (size < 1pt)
    *   **Faux-Bold Merging**: Detects and merges double-rendered text used to simulate bolding
    *   **Re-Distillation**: For rotated text, creates temporary "straight" PDF fragments and re-extracts for perfect accuracy
    *   **Malformed Quote Fixing**: Repairs common PDF encoding issues with quotation marks

6.  **Semantic Grouping**: The `TextGrouping` utility analyzes gaps and alignment to build a hierarchical `Section` model with:
    - Adaptive gap calculation (font-size relative)
    - Baseline alignment detection
    - List and TOC detection
    - Indentation level computation
    - Margin calculation for reflowable layout

**Performance:**
- Average: 85ms per page
- Gap calculation: O(1)
- Text grouping: O(n log n) with spatial indexing
- Character processing: O(n) with minimal allocations

### Image Extraction

Image extraction uses a sophisticated dual-strategy approach that combines stream-based positioning with metadata mining.

**Architecture:**
- **Engine**: `engine/image_processor.py` - Coordinates image extraction workflow
- **Graphics State**: `processors/pdf_graphics.py` - Tracks CTM and graphics state throughout rendering
- **Transforms**: `utils/pdf_transforms.py` - Matrix decomposition for rotation/scale/skew

**Key Features:**

1.  **Parallel Execution**: Image extraction runs concurrently with text extraction for optimal performance (via `asyncio.gather`).

2.  **Stream Processing & CTM Tracking**: 
    - `GraphicsStateTracker` simulates the PDF rendering engine
    - Maintains Current Transformation Matrix (CTM) through all operations
    - Tracks graphics state (opacity, fill/stroke alpha, clip paths)
    - Handles nested form XObjects with stack-based context

3.  **Intelligent Merging**: The system:
    - Mines "Rich Image" data (original bytes, format, metadata) from PDF resources
    - Extracts "Placeholder" stream data with precise transform information
    - Merges both to provide high-fidelity assets with accurate positioning

4.  **Format Handling**:
    - Supports multiple PDF image formats (JPEG, PNG, TIFF)
    - Handles compression filters (FlateDecode, DCTDecode, etc.)
    - ICC profile embedding for color accuracy
    - SMask transparency for PNG output
    - Base64 encoding for web-friendly embedding

5.  **Coordinate Transformation**:
    - Image unit square mapping: (0,0) to (1,1) → final position
    - CTM application for scaling, rotation, and skew
    - Conversion to Y-down coordinates for web compatibility
    - Precision: 2 decimal places (configurable)

6.  **Resource Management**:
    - Image deduplication via SHA-256 hashing
    - Optional caching for performance
    - Reference counting for shared resources
    - Form XObject recursion with cycle detection

### PDF Modification

The modification process rebuilds the PDF's content stream with surgical precision, removing unwanted elements while preserving document integrity.

**Architecture:**
- **Engine**: `engine/content_modifier.py` - Coordinates content removal and PDF regeneration
- **Stream Parsing**: Processes PDF operators while tracking graphics state
- **Resource Management**: Reference counting for shared XObject resources

**Key Features:**

1.  **State-Aware Parsing**: 
    - Parses content stream while maintaining graphics state context
    - Tracks CTM for accurate bounding box calculation
    - Handles nested form XObjects and save/restore operations

2.  **Content Filtering**: 
    - Text removal: Intersection over Union (IoU) matching with configurable threshold (0.8 default)
    - Image removal: Name-based matching with bbox verification
    - Maintains operator sequence integrity
    - Preserves graphics state consistency

3.  **Stream Rebuilding**: 
    - Filters operators corresponding to removed elements
    - Writes remaining operators to new, clean content stream
    - Preserves PDF structure (resources, annotations, metadata)

4.  **Resource Management**: 
    - Reference counting: Tracks image XObject usage across pages
    - Smart cleanup: Only removes resources with zero references
    - Handles shared resources: Prevents accidental deletion of images used elsewhere
    - Form XObject handling: Processes nested content streams

5.  **PDF Assembly**: 
    - Uses `pikepdf` for final PDF generation
    - Maintains document structure and metadata
    - Preserves page-level attributes (rotation, media box, etc.)
    - Ensures valid PDF/A compliance where applicable

**Configuration:**
```python
ContentModifierOptions(
    reference_counting=True,      # Track resource references
    overlap_threshold=0.8,        # IoU threshold for text matching
    text_tolerance=2.0,           # Pixel tolerance for bboxes
    preserve_empty_content=False  # Remove empty streams
)
```

### Reliability & Hygiene

-   **Fast-Fail Validation**: Before heavy processing begins, files are validated for encryption, corruption, and size limits to protect server resources.
-   **Resource Safety**: Critical operations are wrapped in Python Context Managers to ensure file handles are strictly closed, preventing memory leaks and OS-level file locks.
-   **CSS Translation**: Internal PDF font names are mapped to standard CSS properties (`fontFamily`, `fontWeight`, `fontStyle`) for clean frontend usage.

## API Endpoints

### Recommended: Unified Endpoint

For optimal performance, use the `/process-pdf` endpoint for all batch processing operations.

**Why use the unified endpoint?**
-   **Parallel Execution**: Uses `asyncio.gather` to run heavy text and image extraction tasks concurrently.
-   **~3x faster**: Combines extraction, layout analysis, and content removal in a single optimized operation.
-   **Reduced network overhead**: 1 API call instead of 2+ separate calls
-   **Atomic operation**: Guarantees consistency between extracted data and the modified PDF
-   **Simplified client logic**: Single request/response cycle instead of orchestrating multiple calls

**All endpoints process every page in the uploaded PDF file.** The client is responsible for creating PDFs containing only the pages it wants to process.

### POST /process-pdf

A unified endpoint that performs text extraction, image extraction, and content removal in a single, efficient operation. This is the **recommended endpoint** for all page processing.

-   **Request Body**: `multipart/form-data`
    -   `file`: The uploaded PDF file (can contain one or more pages).
    -   `config`: A JSON string representing the `ProcessPdfConfig` object, which specifies options for text and image extraction.

-   **Example `config` JSON string**:
    ```json
    {
      "text_config": {
        "enable_text_grouping": true,
        "enable_list_detection": true
      },
      "image_config": {
        "include_image_data": true
      }
    }
    ```

-   **Success Response (200)**: `application/json`
    -   A `ProcessPdfResponse` object containing an array of element lists (one per page) and the complete, modified PDF as a Base64-encoded string.

    ```json
    {
      "processedPages": [
        [
          /* ... text and image elements for page 1 ... */
        ],
        [
          /* ... text and image elements for page 2 ... */
        ]
      ],
      "modifiedPdf": "JVBERi0xLjcKJeLjz9MKMSAwIG9iago8PC9..."
    }
    ```

### Individual Endpoints (Specialized Use Cases)

The following endpoints are maintained for specific scenarios where you need only one type of operation (e.g., extraction without modification). However, they are less efficient than the unified endpoint for typical workflows.

### POST /extract-pdf-text

Extracts all structured text from all pages of an uploaded PDF, groups it into logical sections (paragraphs, lists, etc.), and returns a structured representation as an array of element lists.

-   **Request Body**: `multipart/form-data`
    -   `file`: The uploaded PDF file (can contain one or more pages).
    -   Optional `Form` fields for configuration (e.g., `enable_text_grouping`, `max_horizontal_gap`, etc.).

-   **Success Response (200)**: `application/json`
    -   An array of element lists. Each inner array contains the text elements for a single page, in page order.

     ```json
    [
      [
        {
          "id": "bf2c3c5c...",
          "rotation": 0,
          "x": 56.7,
          "y": 72.9,
          "width": 285,
          "height": 35,
          "type": "text",
          "text": "A Heading Example",
          "sections": [
            {
              "indentationLevel": 0,
              "sectionId": "section-17633...",
              "lines": [
                [
                  {
                    "text": "A",
                    "fontFamily": "Helvetica-Bold",
                    "fontWeight": "normal",
                    "fontStyle": "normal",
                    "fontSize": 14,
                    "color": "#000000",
                    "underline": false,
                    "strikethrough": false,
                    "originalFontName": "Helvetica-Bold"
                  },
                  {
                    "text": "Heading",
                    "fontFamily": "Helvetica-Bold",
                    "fontWeight": "bold",
                    "fontStyle": "normal",
                    "fontSize": 14,
                    "color": "#000000",
                    "underline": false,
                    "strikethrough": false,
                    "originalFontName": "Helvetica-Bold"
                  },
                  {
                    "text": "Example",
                    "fontFamily": "Helvetica-Bold",
                    "fontWeight": "normal",
                    "fontStyle": "normal",
                    "fontSize": 14,
                    "color": "#000000",
                    "underline": false,
                    "strikethrough": false,
                    "originalFontName": "Helvetica-Bold"
                  }
                ]
              ],
              "lineHeight": 1.2,
              "marginBottom": 10,
              "textAlign": "left"
            }
          ]
        }
      ]
    ]
    ```

### POST /extract-pdf-images

Extracts metadata for all images from all pages of an uploaded PDF.

-   **Request Body**: `multipart/form-data`
    -   `file`: The uploaded PDF file (can contain one or more pages).
    -   Optional `Form` fields for image extraction configuration.

-   **Success Response (200)**: `application/json`
    -   An array of element lists. Each inner array contains only the image elements for that page, in page order.

    ```json
    [
      [
        {
          "id": "img-0",
          "rotation": 0,
          "x": 100,
          "y": 200,
          "width": 200,
          "height": 150,
          "type": "image",
          "name": "Im1",
          "imageIndex": 0,
          "mimeType": "image/jpeg",
          "format": "JPEG",
          "skewX": 0,
          "skewY": 0
        }
      ]
    ]
    ```

### POST /extract-image

Extracts the full data for a single image, identified by its resource name (`name`) and page number.

-   **Request Body**: `multipart/form-data`
    -   `file`: The uploaded PDF file (typically a single-page PDF).
    -   `page_number`: The 1-based page number (usually 1 for single-page PDFs).
    -   `image_id`: The PDF XObject name of the image (e.g., "Im1").

-   **Success Response (200)**: Binary image data
    -   Returns the raw image bytes with the appropriate MIME type.

### POST /remove-pdf-content

Removes specified content from one or more pages of a PDF and returns the complete modified PDF file.

-   **Request Body**: `multipart/form-data`
    -   `file`: The original PDF file.
    -   `removal_data`: A JSON string representing a dictionary where keys are page numbers (as strings) and values are removal specifications.

-   **Example `removal_data` JSON string**:
    ```json
    {
      "1": {
        "text_bboxes": [
          { "x": 50, "y": 50, "width": 100, "height": 20 }
        ],
        "image_bboxes": [
          {
            "name": "Im1",
            "bbox": { "x": 100, "y": 200, "width": 200, "height": 150 }
          }
        ]
      },
      "3": {
        "text_bboxes": [
          { "x": 75, "y": 100, "width": 150, "height": 25 }
        ]
      }
    }
    ``

-   **Success Response (200)**: `application/pdf`
    -   The raw binary data of the complete modified PDF file containing all pages.

## Documentation

The [docs/](docs/) directory contains comprehensive documentation:
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Design patterns and architectural decisions
- **[IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)**: Critical implementation details and non-obvious solutions

## Setup and Running

1.  **Navigate to the directory**:
    ```bash
    cd py-server
    ```

2.  **Create the Virtual Environment**:
    **Important:** You must name the folder `venv`. The `start_server.sh` script is configured to look specifically for this directory.
    ```bash
    python3 -m venv venv
    ```

3.  **Install Dependencies**:
    Install the libraries directly into the virtual environment using its specific `pip` executable.
    ```bash
    ./venv/bin/pip install -r requirements.txt
    ```

4.  **Run the Server**:
    Use the startup script to launch the application. This script automatically handles port cleanup and points to the correct Python executable inside your `venv`.
    ```bash
    ./start_server.sh
    ```