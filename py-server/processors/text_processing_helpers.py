"""
Text Processing Helper Classes and Functions

Helper classes and utilities for text extraction and processing.
Extracted from the legacy pdf_engine module.
"""

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.pdf_types import ProcessingTextRun
from utils.font_mapping import (
    convert_color_to_rgb,
    convert_rgb_to_hex,
    get_font_weight_and_style,
    map_pdf_font_to_css,
)
from utils.pdf_transforms import decompose_ctm

logger = logging.getLogger(__name__)

COORDINATE_PRECISION = 2
ROTATION_THRESHOLD_DEGREES = 1.0


@dataclass
class PdfRotationMarker:
    """Placeholder for rotated text position in stream order."""
    page_num: int
    marker_id: int
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    
    def __repr__(self):
        return f"<RotationMarker id={self.marker_id} angle={self.rotation}° at ({self.x:.1f}, {self.y:.1f})>"


@dataclass
class StyledRun:
    """Contiguous run of text with consistent styling properties."""
    text: str
    font_name: str
    font_size: float
    color: Optional[Any]
    x0: float
    top: float
    x1: float
    bottom: float
    chars: List[Dict]
    rotation: float = 0.0
    tm_start_x: Optional[float] = None  # Text matrix X position at start of render_string
    tm_end_x: Optional[float] = None    # Text matrix X position at end of render_string


def _get_char_orientation(char: Dict) -> float:
    """Calculate character rotation using decompose_ctm."""
    matrix = char.get("matrix")
    if matrix and len(matrix) >= 6:
        transformation = decompose_ctm(list(matrix))
        angle = round(transformation.rotation, 2)
        
        abs_angle = abs(angle)
        if abs_angle >= 1.0:
            is_orthogonal = any(abs(abs_angle - orth) < 1.0 for orth in [0, 90, 180, 270, 360])
            if not is_orthogonal:
                logger.debug(f"Non-orthogonal rotation detected: {angle}° from matrix {matrix}")
        return angle
    return 0.0


def _is_rotated_char(char: Dict) -> bool:
    """Check if character exceeds rotation threshold."""
    char_orientation = _get_char_orientation(char)
    return abs(char_orientation) >= ROTATION_THRESHOLD_DEGREES


def create_processing_text_run(run: StyledRun) -> ProcessingTextRun:
    """Convert StyledRun to ProcessingTextRun with font mapping and color conversion."""
    font_name = run.font_name
    font_weight, font_style = get_font_weight_and_style(font_name)
    text_color = convert_rgb_to_hex(convert_color_to_rgb(run.color))
    
    # Fix malformed quotes from incorrect font encodings
    fixed_text = _fix_malformed_quotes(run.text)
    
    element_height = run.bottom - run.top
    element_width = run.x1 - run.x0
    
    return ProcessingTextRun(
        text=fixed_text,
        rotation=run.rotation,
        x=round(run.x0, COORDINATE_PRECISION),
        y=round(run.top, COORDINATE_PRECISION),
        width=round(element_width, COORDINATE_PRECISION),
        height=round(element_height, COORDINATE_PRECISION),
        fontFamily=map_pdf_font_to_css(font_name),
        fontSize=round(run.font_size, COORDINATE_PRECISION),
        fontWeight=font_weight,
        fontStyle=font_style,
        color=text_color,
        textDecoration="none",
        originalFontName=font_name,
        tm_start_x=run.tm_start_x,
        tm_end_x=run.tm_end_x
    )


def _normalize_font_name(fontname: str) -> str:
    """Normalize font name by removing PDF subset prefixes.
    
    Removes random 6-character prefixes and subset markers (+)
    to allow fonts from different subsets to be treated as identical.
    """
    if not fontname:
        return ""
    
    # Remove common subset prefix patterns (6 chars + +)
    if '+' in fontname:
        fontname = fontname.split('+', 1)[1]
    
    # Remove PDF subset prefixes (random 6-character prefix before actual font name)
    # Pattern: starts with 6 letters (any case) followed by an uppercase letter
    # Examples: PdbpbbLato, DrhchpLato, AbcdefHelvetica
    match = re.match(r'^[A-Za-z]{6}([A-Z].*)$', fontname)
    if match:
        fontname = match.group(1)
    
    return fontname


def _fix_malformed_quotes(text: str) -> str:
    """Fix malformed quote characters from incorrect font encodings.
    
    Corrects patterns where / = " and 0 = " in poorly encoded fonts.
    """
    if not text:
        return text
    
    # Pattern 1: /word0 -> "word"
    text = re.sub(r'/([A-Za-z][A-Za-z\s,.-]*?)0', r'"\1"', text)
    
    # Pattern 2: Handle multi-word spans with newlines
    text = re.sub(r'/([A-Za-z][A-Za-z\s\n,.-]{0,50}?)0(?=[\s,.)!?;:\n]|$)', r'"\1"', text)
    
    return text


def _normalize_text_content(text: str) -> str:
    """Normalize text by fixing malformed quotes and replacing special characters."""
    if not text:
        return text
    
    # First, fix malformed quotes from incorrectly encoded fonts
    text = _fix_malformed_quotes(text)
    
    # Then map special Unicode characters to their standard equivalents
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...', # Horizontal ellipsis
        '\xa0': ' ',    # Non-breaking space
    }
    
    for special_char, replacement in replacements.items():
        text = text.replace(special_char, replacement)
    
    return text


class TextRunAccumulator:
    """Stateful accumulator for building StyledRuns from characters."""
    
    def __init__(self, skip_rotation_check: bool = False):
        self.current_run_data: Optional[Dict] = None
        self.completed_runs: List[StyledRun] = []
        self.skip_rotation_check = skip_rotation_check

    def add(self, char: Dict) -> Optional[StyledRun]:
        """
        Add a character and return a completed run if styling changed.
        
        This is the streaming API used by StreamOrderDevice.
        Returns a StyledRun when style changes force finalization.
        """
        self.process_char(char)
        # If process_char started a new run, a completed run was added
        if len(self.completed_runs) > 0:
            # Return and remove the last completed run
            return self.completed_runs.pop(0)
        return None

    def process_char(self, char: Dict):
        """Process a single character, accumulating runs or starting new ones."""
        fontname = char.get("fontname")
        size = char.get("size")
        color = char.get("non_stroking_color") or char.get("ncs")
        x0 = char.get("x0")
        x1 = char.get("x1")
        top = char.get("top")
        bottom = char.get("bottom")
        text = char.get("text")

        if not all([fontname, size is not None, x0 is not None, top is not None]):
            return

        normalized_fontname = _normalize_font_name(fontname)
        
        # Check if character is rotated (if not skipping rotation check)
        is_rotated = False if self.skip_rotation_check else _is_rotated_char(char)
        
        # If this character is rotated, finalize current run and skip it
        if is_rotated:
            if self.current_run_data:
                self._finalize_current_run()
            return

        # Check if we should continue the current run or start a new one
        should_start_new_run = (
            self.current_run_data is None
            or self.current_run_data["font_name"] != normalized_fontname
            or abs(self.current_run_data["font_size"] - size) > 0.01
            or self.current_run_data["color"] != color
            or abs(self.current_run_data["top"] - top) > 0.5
            or abs(self.current_run_data["bottom"] - bottom) > 0.5
        )

        if should_start_new_run:
            if self.current_run_data:
                self._finalize_current_run()
            
            self.current_run_data = {
                "text": text,
                "font_name": normalized_fontname,
                "font_size": size,
                "color": color,
                "x0": x0,
                "top": top,
                "x1": x1,
                "bottom": bottom,
                "chars": [char],
                "rotation": 0.0,
                "tm_start_x": None,  # Will be set by render_string
                "tm_end_x": None
            }
        else:
            # Continue current run
            self.current_run_data["text"] += text
            self.current_run_data["x1"] = x1
            self.current_run_data["chars"].append(char)

    def _finalize_current_run(self):
        """Convert current run data to StyledRun and add to completed runs."""
        if self.current_run_data:
            run = StyledRun(
                text=self.current_run_data["text"],
                font_name=self.current_run_data["font_name"],
                font_size=self.current_run_data["font_size"],
                color=self.current_run_data["color"],
                x0=self.current_run_data["x0"],
                top=self.current_run_data["top"],
                x1=self.current_run_data["x1"],
                bottom=self.current_run_data["bottom"],
                chars=self.current_run_data["chars"],
                rotation=self.current_run_data["rotation"],
                tm_start_x=self.current_run_data.get("tm_start_x"),
                tm_end_x=self.current_run_data.get("tm_end_x")
            )
            self.completed_runs.append(run)
            self.current_run_data = None

    def finalize(self) -> List[StyledRun]:
        """Finalize any remaining run and return all completed runs."""
        if self.current_run_data:
            self._finalize_current_run()
        result = self.completed_runs
        self.completed_runs = []  # Clear the list after returning
        return result
    
    def flush(self) -> List[StyledRun]:
        """Flush accumulated runs and clear the accumulator."""
        return self.finalize()
