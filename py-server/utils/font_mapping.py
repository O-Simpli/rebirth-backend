"""
Font mapping utilities for PDF Content Extraction
Maps PDF fonts to CSS font families with comprehensive coverage
"""

import re
from functools import lru_cache
from typing import Tuple

@lru_cache(maxsize=128)
def map_pdf_font_to_css(font_name: str) -> str:
    """
    Map PDF font names to CSS font families with comprehensive mapping
    Matches the TypeScript implementation exactly
    """
    if not font_name:
        return "serif"
    
    base_name = font_name.split(',')[0].strip().strip('\'"')
    base_name = re.sub(r'^[A-Z]{6}\+', '', base_name)
    base_name = re.sub(r'^[A-Za-z]{6}(?=[A-Z])', '', base_name)
    
    base_name = re.sub(r'_\d+wght', '', base_name, flags=re.IGNORECASE)
    base_name = re.sub(r'_opsz\d+', '', base_name, flags=re.IGNORECASE)
    base_name = re.sub(r'[_-]?\d+px', '', base_name, flags=re.IGNORECASE)
    
    base_name = re.sub(r'-(Bold|Italic|BoldItalic|Regular|Normal|MT|PS)$', '', base_name, flags=re.IGNORECASE)
    base_name = re.sub(r',?(Bold|Italic|BoldItalic|Regular|Normal)$', '', base_name, flags=re.IGNORECASE)
    base_name = re.sub(r'(?:[-_\s]|(?<=[a-z]))(Thin|Light|ExtraLight|UltraLight|Medium|SemiBold|DemiBold|ExtraBold|UltraBold|Black|Heavy)$', '', base_name, flags=re.IGNORECASE)
    
    clean_name = base_name.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    # Font mapping organized by categories - matches TypeScript implementation
    font_map = {
        # Times family
        'times': 'Times New Roman, Times, serif',
        'timesnewroman': 'Times New Roman, Times, serif',
        'timesroman': 'Times New Roman, Times, serif',
        
        # Helvetica/Arial family
        'helvetica': 'Helvetica, Arial, sans-serif',
        'arial': 'Arial, Helvetica, sans-serif',
        'arialmt': 'Arial, Helvetica, sans-serif',
        
        # Liberation fonts (open source alternatives)
        'liberationsans': 'Liberation Sans, Arial, Helvetica, sans-serif',
        'liberationserif': 'Liberation Serif, Times, serif',
        'liberationmono': 'Liberation Mono, Courier, monospace',
        
        # DejaVu fonts (common in Linux PDFs)
        'dejavusans': 'DejaVu Sans, Arial, Helvetica, sans-serif',
        'dejavuserif': 'DejaVu Serif, Times, serif',
        'dejavusansmono': 'DejaVu Sans Mono, Courier, monospace',
        
        # Modern web fonts
        'opensans': 'Open Sans, Arial, Helvetica, sans-serif',
        'sourcesanspro': 'Source Sans Pro, Arial, Helvetica, sans-serif',
        'sourceserifpro': 'Source Serif Pro, Times, serif',
        
        # Courier family
        'courier': 'Courier New, Courier, monospace',
        'couriernew': 'Courier New, Courier, monospace',
        
        # System fonts
        'calibri': 'Calibri, sans-serif',
        'verdana': 'Verdana, sans-serif',
        'georgia': 'Georgia, serif',
        'trebuchet': 'Trebuchet MS, sans-serif',
        
        # Special fonts
        'symbol': 'Symbol, serif',
        'zapfdingbats': 'Zapf Dingbats, serif',
        'wingdings': 'Wingdings, serif',
    }
    
    # Check for exact matches
    for pdf_font, css_font in font_map.items():
        if pdf_font in clean_name:
            return css_font
    
    # Handle generic CSS font families
    generic_fonts = ['serif', 'sansserif', 'monospace', 'cursive', 'fantasy']
    if clean_name in generic_fonts:
        return 'sans-serif' if clean_name == 'sansserif' else clean_name
    
    # For unknown fonts, use cleaned name if it looks like a real font
    if base_name and len(base_name) > 1 and not re.match(r'^[a-z]_[a-z0-9_]+$', base_name):
        return f'{base_name}, sans-serif'
    
    return "serif"  # Safe fallback

@lru_cache(maxsize=128)
def get_font_weight_and_style(font_name: str) -> Tuple[str, str]:
    """
    Extract font weight and style from font name
    Matches TypeScript implementation logic
    """
    font_weight = "normal"
    font_style = "normal"
    
    if not font_name:
        return font_weight, font_style
    
    font_name_lower = font_name.lower()
    
    weight_match = re.search(r'(\d{3})(?:wght)?', font_name_lower)
    if weight_match:
        weight_val = int(weight_match.group(1))
        if 100 <= weight_val <= 900 and weight_val % 100 == 0:
            font_weight = str(weight_val)
    
    if font_weight == "normal":
        if any(bold_indicator in font_name_lower for bold_indicator in ['bold', 'black', 'heavy', 'extra', 'ultra']):
            font_weight = "bold"
        elif any(light_indicator in font_name_lower for light_indicator in ['light', 'thin', 'hairline']):
            font_weight = "300"
        elif 'medium' in font_name_lower:
            font_weight = "500"
        elif 'semibold' in font_name_lower:
            font_weight = "600"
    
    # Check for italic - matches TypeScript parseFontStyle
    if any(italic_indicator in font_name_lower for italic_indicator in ['italic', 'oblique', 'slant']):
        font_style = "italic"
    
    return font_weight, font_style

def convert_color_to_rgb(color_info) -> str:
    """
    Convert PDF color information to CSS RGB string
    Handles different color formats (RGB, CMYK, grayscale)
    """
    if not color_info:
        return "rgb(0, 0, 0)"  # Default black
    
    try:
        if isinstance(color_info, (list, tuple)):
            if len(color_info) == 1:
                # Grayscale
                gray = float(color_info[0])
                rgb_val = int(gray * 255)
                return f"rgb({rgb_val}, {rgb_val}, {rgb_val})"
            elif len(color_info) == 3:
                # RGB
                r, g, b = [int(float(c) * 255) for c in color_info]
                return f"rgb({r}, {g}, {b})"
            elif len(color_info) == 4:
                # CMYK - convert to RGB (simplified conversion)
                c, m, y, k = [float(x) for x in color_info]
                r = int(255 * (1 - c) * (1 - k))
                g = int(255 * (1 - m) * (1 - k))
                b = int(255 * (1 - y) * (1 - k))
                return f"rgb({r}, {g}, {b})"
        
        # If it's a single number, treat as grayscale
        if isinstance(color_info, (int, float)):
            rgb_val = int(float(color_info) * 255)
            return f"rgb({rgb_val}, {rgb_val}, {rgb_val})"
            
    except (ValueError, TypeError):
        pass
    
    return "rgb(0, 0, 0)"  # Default black if conversion fails

def convert_rgb_to_hex(rgb_str: str) -> str:
    """
    Convert a CSS rgb() string to hex color.
    Example: "rgb(255, 0, 128)" -> "#ff0080"
    """
    match = re.match(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", rgb_str)
    if not match:
        return "#000000"
    r, g, b = (int(match.group(i)) for i in range(1, 4))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)