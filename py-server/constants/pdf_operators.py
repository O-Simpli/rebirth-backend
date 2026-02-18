"""
PDF Operator Constants

Centralized definitions of all PDF content stream operators used throughout the codebase.
Organized by functional category according to PDF specification.

Reference: PDF 32000-1:2008 specification, Appendix A
"""

# ==============================================================================
# Graphics State Operators (PDF spec 8.4.4)
# ==============================================================================
OP_SAVE_STATE = b'q'                 # Save graphics state
OP_RESTORE_STATE = b'Q'              # Restore graphics state
OP_CTM = b'cm'                       # Modify current transformation matrix
OP_SET_LINE_WIDTH = b'w'             # Set line width
OP_SET_LINE_CAP = b'J'               # Set line cap style
OP_SET_LINE_JOIN = b'j'              # Set line join style
OP_SET_MITER_LIMIT = b'M'            # Set miter limit
OP_SET_DASH = b'd'                   # Set line dash pattern
OP_SET_FLATNESS = b'i'               # Set flatness tolerance
OP_SET_GRAPHICS_STATE_PARAMS = b'gs' # Set parameters from graphics state parameter dict

GRAPHICS_STATE_OPS = {
    OP_SAVE_STATE, OP_RESTORE_STATE, OP_CTM, OP_SET_LINE_WIDTH, 
    OP_SET_LINE_CAP, OP_SET_LINE_JOIN, OP_SET_MITER_LIMIT, 
    OP_SET_DASH, OP_SET_FLATNESS, OP_SET_GRAPHICS_STATE_PARAMS
}

# ==============================================================================
# Color Operators (PDF spec 8.6.8)
# ==============================================================================
# Stroke
OP_SET_GRAY_STROKE = b'G'            # Set Gray color for stroking
OP_SET_RGB_COLOR_STROKE = b'RG'      # Set RGB color for stroking
OP_SET_CMYK_COLOR_STROKE = b'K'      # Set CMYK color for stroking
OP_SET_COLOR_STROKE = b'SC'          # Set color for stroking (general)
OP_SET_COLOR_STROKE_N = b'SCN'       # Set color for stroking (general + name)
OP_SET_COLOR_SPACE_STROKE = b'CS'    # Set color space for stroking

# Fill (Non-Stroke)
OP_SET_GRAY_FILL = b'g'              # Set Gray color for non-stroking
OP_SET_RGB_COLOR_FILL = b'rg'        # Set RGB color for non-stroking
OP_SET_CMYK_COLOR_FILL = b'k'        # Set CMYK color for non-stroking
OP_SET_COLOR_FILL = b'sc'            # Set color for non-stroking (general)
OP_SET_COLOR_FILL_N = b'scn'         # Set color for non-stroking (general + name)
OP_SET_COLOR_SPACE_FILL = b'cs'      # Set color space for non-stroking

COLOR_OPS = {
    OP_SET_GRAY_STROKE, OP_SET_RGB_COLOR_STROKE, OP_SET_CMYK_COLOR_STROKE,
    OP_SET_COLOR_STROKE, OP_SET_COLOR_STROKE_N, OP_SET_COLOR_SPACE_STROKE,
    OP_SET_GRAY_FILL, OP_SET_RGB_COLOR_FILL, OP_SET_CMYK_COLOR_FILL,
    OP_SET_COLOR_FILL, OP_SET_COLOR_FILL_N, OP_SET_COLOR_SPACE_FILL
}

# ==============================================================================
# Text State Operators (PDF spec 9.3)
# ==============================================================================
OP_BEGIN_TEXT = b'BT'            # Begin text object
OP_END_TEXT = b'ET'              # End text object
OP_SET_FONT = b'Tf'              # Set text font and size
OP_SET_CHAR_SPACING = b'Tc'      # Set character spacing
OP_SET_WORD_SPACING = b'Tw'      # Set word spacing
OP_SET_HORIZ_SCALING = b'Tz'     # Set horizontal text scaling
OP_SET_LEADING = b'TL'           # Set text leading
OP_SET_TEXT_RISE = b'Ts'         # Set text rise
OP_SET_TEXT_RENDER = b'Tr'       # Set text rendering mode

TEXT_STATE_OPS = {
    OP_BEGIN_TEXT, OP_END_TEXT, OP_SET_FONT, OP_SET_CHAR_SPACING,
    OP_SET_WORD_SPACING, OP_SET_HORIZ_SCALING, OP_SET_LEADING,
    OP_SET_TEXT_RISE, OP_SET_TEXT_RENDER
}

# ==============================================================================
# Text Positioning Operators (PDF spec 9.4.2)
# ==============================================================================
OP_MOVE_TEXT = b'Td'              # Move text position
OP_MOVE_TEXT_SET_LEADING = b'TD'  # Move text position and set leading
OP_SET_TEXT_MATRIX = b'Tm'        # Set text matrix and text line matrix
OP_NEXT_LINE = b'T*'              # Move to start of next text line

# ==============================================================================
# Text Showing Operators (PDF spec 9.4.3)
# ==============================================================================
OP_SHOW_TEXT = b'Tj'              # Show a text string
OP_SHOW_TEXT_ARRAY = b'TJ'        # Show text strings with positioning
OP_NEXT_LINE_SHOW_TEXT = b"'"     # Move to next line and show text
OP_SET_SPACING_SHOW_TEXT = b'"'   # Set spacing, move to next line, show text

TEXT_SHOWING_OPS = {OP_SHOW_TEXT, OP_SHOW_TEXT_ARRAY, OP_NEXT_LINE_SHOW_TEXT, OP_SET_SPACING_SHOW_TEXT}

# ==============================================================================
# XObject Operators (PDF spec 8.8)
# ==============================================================================
OP_DO_XOBJECT = b'Do'     # Invoke named XObject (image, form, etc.)

XOBJECT_OPS = {OP_DO_XOBJECT}

# ==============================================================================
# Path Construction Operators (PDF spec 8.5.2)
# ==============================================================================
OP_MOVETO = b'm'          # Begin new subpath (moveto)
OP_LINETO = b'l'          # Append straight line segment (lineto)
OP_CURVETO = b'c'         # Append cubic Bézier curve
OP_CURVETO_V = b'v'       # Append cubic Bézier curve (initial point replicated)
OP_CURVETO_Y = b'y'       # Append cubic Bézier curve (final point replicated)
OP_RECTANGLE = b're'      # Append rectangle
OP_CLOSEPATH = b'h'       # Close current subpath

PATH_CONSTRUCTION_OPS = {OP_MOVETO, OP_LINETO, OP_CURVETO, OP_CURVETO_V, OP_CURVETO_Y, OP_RECTANGLE, OP_CLOSEPATH}

# ==============================================================================
# Path Painting Operators (PDF spec 8.5.3)
# ==============================================================================
# Painting
OP_STROKE = b'S'
OP_CLOSE_STROKE = b's'
OP_FILL = b'f'
OP_FILL_OBSOLETE = b'F'
OP_FILL_EVEN_ODD = b'f*'
OP_FILL_STROKE = b'B'
OP_FILL_STROKE_EVEN_ODD = b'B*'
OP_CLOSE_FILL_STROKE = b'b'
OP_CLOSE_FILL_STROKE_EVEN_ODD = b'b*'
OP_END_PATH = b'n'

PATH_PAINTING_OPS = {
    OP_STROKE, OP_CLOSE_STROKE, OP_FILL, OP_FILL_OBSOLETE, OP_FILL_EVEN_ODD,
    OP_FILL_STROKE, OP_FILL_STROKE_EVEN_ODD, OP_CLOSE_FILL_STROKE,
    OP_CLOSE_FILL_STROKE_EVEN_ODD, OP_END_PATH
}

# Sub-groups for logic checks
STROKE_OPS = {
    OP_STROKE, OP_CLOSE_STROKE, OP_FILL_STROKE, OP_FILL_STROKE_EVEN_ODD,
    OP_CLOSE_FILL_STROKE, OP_CLOSE_FILL_STROKE_EVEN_ODD
}

FILL_OPS = {
    OP_FILL, OP_FILL_OBSOLETE, OP_FILL_EVEN_ODD, OP_FILL_STROKE,
    OP_FILL_STROKE_EVEN_ODD, OP_CLOSE_FILL_STROKE, OP_CLOSE_FILL_STROKE_EVEN_ODD
}

EVEN_ODD_OPS = {
    OP_FILL_EVEN_ODD, OP_FILL_STROKE_EVEN_ODD, OP_CLOSE_FILL_STROKE_EVEN_ODD
}

# ==============================================================================
# Clipping Path Operators (PDF spec 8.5.4)
# ==============================================================================
OP_CLIP = b'W'            # Set clipping path using nonzero winding number rule
OP_CLIP_EVEN_ODD = b'W*'  # Set clipping path using even-odd rule

CLIPPING_OPS = {OP_CLIP, OP_CLIP_EVEN_ODD}

# ==============================================================================
# Shading Operators (PDF spec 8.7.4)
# ==============================================================================
OP_SHADING = b'sh'    # Paint area with shading pattern
SHADING_OPS = {OP_SHADING}

# ==============================================================================
# Marked Content Operators (PDF spec 10.5)
# ==============================================================================
OP_MP = b'MP'    # Define marked-content point
OP_DP = b'DP'    # Define marked-content point with property list
OP_BMC = b'BMC'  # Begin marked-content sequence
OP_BDC = b'BDC'  # Begin marked-content sequence with property list
OP_EMC = b'EMC'  # End marked-content sequence

MARKED_CONTENT_OPS = {OP_MP, OP_DP, OP_BMC, OP_BDC, OP_EMC}

# ==============================================================================
# Composite Operator Groups
# ==============================================================================

# All vector graphics operators (paths + painting + shading)
VECTOR_GRAPHICS_OPS = PATH_CONSTRUCTION_OPS | PATH_PAINTING_OPS | SHADING_OPS

# All operators that modify graphics state
STATE_MODIFICATION_OPS = GRAPHICS_STATE_OPS | COLOR_OPS

# All operators that render visible content
RENDERING_OPS = TEXT_SHOWING_OPS | PATH_PAINTING_OPS | SHADING_OPS | XOBJECT_OPS