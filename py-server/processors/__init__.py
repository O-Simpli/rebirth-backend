"""
PDF Processing Components

Stateful processors for PDF content extraction and manipulation. These components
maintain internal state and implement complex algorithms:

- StreamOrderDevice: PDFMiner device for stream-order text extraction
- TextGrouper: Semantic text grouping with spatial indexing
- GraphicsStateTracker: CTM and graphics state tracking
- TextRunAccumulator: Character-to-run accumulation with styling
- Position tracking strategies: Regular, rotated, and normalized text positioning
- Text normalizer: Rotated text re-distillation

These differ from utils/ which contains pure, stateless functions.
"""

from processors.stream_order_device import StreamOrderDevice
from processors.text_grouping import TextGrouper, group_text_elements
from processors.pdf_graphics import GraphicsStateTracker
from processors.text_processing_helpers import TextRunAccumulator, PdfRotationMarker
from processors.position_tracking import PositionTrackingStrategy, RegularTextStrategy

__version__ = "2.0.0"
__all__ = [
    'StreamOrderDevice',
    'TextGrouper',
    'group_text_elements',
    'GraphicsStateTracker',
    'TextRunAccumulator',
    'PdfRotationMarker',
    'PositionTrackingStrategy',
    'RegularTextStrategy',
]
