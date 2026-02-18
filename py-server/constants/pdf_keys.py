"""
PDF Dictionary Keys and Name Constants
"""

# Resource Dictionary Keys
KEY_RESOURCES = "/Resources"
KEY_XOBJECT = "/XObject"
KEY_EXT_GSTATE = "/ExtGState"
KEY_PATTERN = "/Pattern"
KEY_SHADING = "/Shading"
KEY_COLOR_SPACE = "/ColorSpace"
KEY_FONT = "/Font"

# Object Types and Subtypes
KEY_TYPE = "/Type"
KEY_SUBTYPE = "/Subtype"
VAL_IMAGE = "/Image"
VAL_FORM = "/Form"
VAL_GROUP = "/Group"

# Image / Form Properties
KEY_WIDTH = "/Width"
KEY_HEIGHT = "/Height"
KEY_MATRIX = "/Matrix"
KEY_BBOX = "/BBox"

# Graphics State Parameter Keys (ExtGState)
KEY_FILL_OPACITY = "/ca"         # Non-stroking alpha constant
KEY_STROKE_OPACITY = "/CA"       # Stroking alpha constant
KEY_BLEND_MODE = "/BM"           # Blend mode
KEY_SOFT_MASK = "/SMask"         # Soft mask
KEY_LINE_WIDTH = "/LW"           # Line width
KEY_LINE_CAP = "/LC"             # Line cap style
KEY_LINE_JOIN = "/LJ"            # Line join style
KEY_MITER_LIMIT = "/ML"          # Miter limit
VAL_NONE = "/None"

# Pattern and Shading Keys
KEY_PATTERN_TYPE = "/PatternType"
KEY_SHADING_TYPE = "/ShadingType"
KEY_COORDS = "/Coords"
KEY_BACKGROUND = "/Background"
KEY_DOMAIN = "/Domain"
KEY_FUNCTION = "/Function"
KEY_FUNCTIONS = "/Functions"
KEY_BOUNDS = "/Bounds"
KEY_C0 = "/C0"
KEY_C1 = "/C1"
KEY_N = "/N"