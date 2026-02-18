"""
PDF Processing Engine

Core engine module for coordinating PDF operations.
Contains the unified PDFEngine class and specialized processors.
"""

__version__ = "2.0.0"

from engine.pdf_engine import PDFEngine
from engine.config import EngineConfig, ProcessorOptions, PageRange
from engine.base_processor import BaseProcessor, ProcessorProtocol
from engine.text_processor import TextProcessor, TextProcessorOptions
from engine.image_processor import ImageProcessor, ImageProcessorOptions
from engine.content_modifier import ContentModifier, ContentModifierOptions

__all__ = [
    'PDFEngine',
    'EngineConfig',
    'ProcessorOptions',
    'PageRange',
    'BaseProcessor',
    'ProcessorProtocol',
    'TextProcessor',
    'TextProcessorOptions',
    'ImageProcessor',
    'ImageProcessorOptions',
    'ContentModifier',
    'ContentModifierOptions',
]
