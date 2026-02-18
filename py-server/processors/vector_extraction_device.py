"""
PDF Vector Extraction Device

Core extraction logic for PDF vector graphics with gradient and pattern support.
"""

import base64
import hashlib
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pikepdf
from pikepdf import parse_content_stream

from constants.pdf_operators import (
    OP_SAVE_STATE, OP_RESTORE_STATE, OP_CTM,
    OP_SET_LINE_WIDTH, OP_SET_LINE_CAP, OP_SET_LINE_JOIN, 
    OP_SET_MITER_LIMIT, OP_SET_DASH, OP_SET_GRAPHICS_STATE_PARAMS,
    OP_SET_RGB_COLOR_STROKE, OP_SET_RGB_COLOR_FILL,
    OP_SET_GRAY_STROKE, OP_SET_GRAY_FILL,
    OP_SET_CMYK_COLOR_STROKE, OP_SET_CMYK_COLOR_FILL,
    OP_SET_COLOR_STROKE, OP_SET_COLOR_STROKE_N,
    OP_SET_COLOR_FILL, OP_SET_COLOR_FILL_N,
    OP_DO_XOBJECT, OP_SHADING,
    OP_CLIP, OP_CLIP_EVEN_ODD,
    OP_END_PATH,
    PATH_CONSTRUCTION_OPS, PATH_PAINTING_OPS,
    STROKE_OPS, FILL_OPS, EVEN_ODD_OPS, MARKED_CONTENT_OPS
)

from constants.pdf_keys import (
    KEY_RESOURCES, KEY_XOBJECT, KEY_EXT_GSTATE, KEY_PATTERN, KEY_SHADING, KEY_COLOR_SPACE,
    KEY_TYPE, KEY_SUBTYPE, VAL_IMAGE, VAL_FORM, VAL_GROUP,
    KEY_WIDTH, KEY_HEIGHT, KEY_MATRIX, KEY_BBOX,
    KEY_FILL_OPACITY, KEY_STROKE_OPACITY, KEY_BLEND_MODE, KEY_SOFT_MASK,
    KEY_LINE_WIDTH, KEY_LINE_CAP, KEY_LINE_JOIN, KEY_MITER_LIMIT, VAL_NONE,
    KEY_PATTERN_TYPE, KEY_SHADING_TYPE, KEY_COORDS, KEY_BACKGROUND,
    KEY_DOMAIN, KEY_FUNCTION, KEY_FUNCTIONS, KEY_BOUNDS, KEY_C0, KEY_C1, KEY_N
)

logger = logging.getLogger(__name__)

# Regular expressions for SVG parsing
_NUM_RE = re.compile(r"-?\d+\.?\d*")
_CMD_RE = re.compile(r"([MLCHVZ])\s*([^MLCHVZ]*)")


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def _b64(data: str) -> str:
    """Base64 encode a string."""
    return base64.b64encode(data.encode("utf-8")).decode("utf-8")


def stable_id(*parts: Any) -> str:
    """Generate stable hash ID from parts."""
    s = "|".join(str(p) for p in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _rgb_to_hex(rgb: Optional[Tuple[float, float, float]]) -> str:
    """Convert RGB tuple to hex string."""
    if rgb is None:
        return "#000000"
    r, g, b = (_clamp01(rgb[0]), _clamp01(rgb[1]), _clamp01(rgb[2]))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def extract_path_data(svg_content: str) -> Optional[str]:
    """Extract SVG path 'd' attribute for direct rendering."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg_content)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        path = root.find('.//svg:path', ns) or root.find('.//path')
        if path is not None:
            return path.get('d')
    except Exception:
        pass
    return None


def _gray_to_hex(g: float) -> str:
    """Convert grayscale value to hex string."""
    g = _clamp01(g)
    v = int(g * 255)
    return f"#{v:02x}{v:02x}{v:02x}"


def _cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
    """Convert CMYK to RGB color space."""
    c, m, y, k = (_clamp01(c), _clamp01(m), _clamp01(y), _clamp01(k))
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return (r, g, b)


def _cmyk_to_hex(c: float, m: float, y: float, k: float) -> str:
    """Convert CMYK to hex string."""
    r, g, b = _cmyk_to_rgb(c, m, y, k)
    return _rgb_to_hex((r, g, b))


def _mat_apply(m: List[float], x: float, y: float) -> Tuple[float, float]:
    """Apply transformation matrix to point."""
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _mat_mul(m1: List[float], m2: List[float]) -> List[float]:
    """Multiply two transformation matrices."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    ]


def _op_str(op: Union[bytes, str]) -> str:
    """Safely convert operator byte constant to string for comparisons."""
    if isinstance(op, bytes):
        return op.decode('latin-1')
    return str(op)

# -------------------------------------------------------------------
# Pre-computed Operator String Sets (Performance Optimization)
# -------------------------------------------------------------------
# Convert byte constants to strings once for O(1) lookup in loops
_PATH_OPS_STR = {op.decode('latin-1') for op in PATH_CONSTRUCTION_OPS}
_PAINT_OPS_STR = {op.decode('latin-1') for op in PATH_PAINTING_OPS}
_STROKE_OPS_STR = {op.decode('latin-1') for op in STROKE_OPS}
_FILL_OPS_STR = {op.decode('latin-1') for op in FILL_OPS}
_EVEN_ODD_OPS_STR = {op.decode('latin-1') for op in EVEN_ODD_OPS}
_END_PATH_STR = OP_END_PATH.decode('latin-1')
_MARKED_CONTENT_OPS_STR = {op.decode('latin-1') for op in MARKED_CONTENT_OPS}


# -------------------------------------------------------------------
# Graphics State
# -------------------------------------------------------------------

class GraphicsState:
    """Tracks PDF graphics state including CTM, colors, line styles, opacity, etc."""
    __slots__ = (
        "ctm",
        "stroke_color",
        "fill_color",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_array",
        "dash_phase",
        "fill_opacity",
        "stroke_opacity",
        "blend_mode",
        "has_soft_mask",
        "fill_pattern",
        "stroke_pattern",
        "fill_color_space",
        "stroke_color_space",
    )

    def __init__(self):
        self.ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.stroke_color: Optional[Tuple[float, float, float]] = None
        self.fill_color: Optional[Tuple[float, float, float]] = None
        self.line_width: float = 1.0
        self.line_cap: int = 0  # 0=butt, 1=round, 2=square
        self.line_join: int = 0  # 0=miter, 1=round, 2=bevel
        self.miter_limit: float = 10.0
        self.dash_array: List[float] = []
        self.dash_phase: float = 0.0
        self.fill_opacity: float = 1.0
        self.stroke_opacity: float = 1.0
        self.blend_mode: Optional[str] = None
        self.has_soft_mask: bool = False
        self.fill_pattern: Optional[dict] = None
        self.stroke_pattern: Optional[dict] = None
        self.fill_color_space = None
        self.stroke_color_space = None

    def copy(self) -> "GraphicsState":
        """Create a deep copy of graphics state for q/Q stack."""
        st = GraphicsState()
        st.ctm = self.ctm[:]
        st.stroke_color = self.stroke_color
        st.fill_color = self.fill_color
        st.line_width = self.line_width
        st.line_cap = self.line_cap
        st.line_join = self.line_join
        st.miter_limit = self.miter_limit
        st.dash_array = self.dash_array[:]
        st.dash_phase = self.dash_phase
        st.fill_opacity = self.fill_opacity
        st.stroke_opacity = self.stroke_opacity
        st.blend_mode = self.blend_mode
        st.has_soft_mask = self.has_soft_mask
        st.fill_pattern = self.fill_pattern
        st.stroke_pattern = self.stroke_pattern
        st.fill_color_space = self.fill_color_space
        st.stroke_color_space = self.stroke_color_space
        return st

    def apply_ctm(self, a, b, c, d, e, f):
        """Apply CTM transformation."""
        self.ctm = _mat_mul([float(a), float(b), float(c), float(d), float(e), float(f)], self.ctm)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform point by current CTM."""
        return _mat_apply(self.ctm, x, y)


# -------------------------------------------------------------------
# Vector Extraction Device
# -------------------------------------------------------------------

class VectorExtractionDevice:
    """PDF vector graphics extraction device."""

    def __init__(self, page_height: float, mediabox_bottom: float = 0.0, include_pattern_image_data: bool = False, generate_svg: bool = True, generate_path_data: bool = True, 
                 split_sparse_vectors: bool = False, sparse_coverage_threshold: float = 20.0, sparse_merge_distance: float = 10.0,
                 enable_overlap_grouping: bool = False, overlap_check_window: int = 20, 
                 overlap_method: str = "area", overlap_threshold: float = 0.1, proximity_threshold: Optional[float] = None,
                 enable_post_processing_merge: bool = False):
        self.page_height = page_height
        self.mediabox_bottom = mediabox_bottom
        self.include_pattern_image_data = include_pattern_image_data
        self.generate_svg = generate_svg
        self.generate_path_data = generate_path_data
        
        # Sparse vector splitting options (stream-based)
        self.split_sparse_vectors = split_sparse_vectors
        self.sparse_coverage_threshold = sparse_coverage_threshold
        self.sparse_merge_distance = sparse_merge_distance
        self.page_area = 612.0 * page_height  # Approximate page area
        
        # Overlap grouping options (stream-based)
        self.enable_overlap_grouping = enable_overlap_grouping
        self.overlap_check_window = overlap_check_window  # Check last N vectors
        self.overlap_method = overlap_method  # "area", "iou", "intersection"
        self.overlap_threshold = overlap_threshold
        self.proximity_threshold = proximity_threshold  # Optional proximity grouping
        self.enable_post_processing_merge = enable_post_processing_merge  # Full O(N²) post-processing
        self.next_group_id = 1  # Counter for assigning group IDs
        
        # Clipping region tracking - True logical grouping
        self.clipping_regions: List[Dict[str, Any]] = []  # All clipping regions encountered
        self.active_clipping_region: Optional[Dict[str, Any]] = None  # Current innermost clipping
        self.clipping_region_id = 0  # Counter for unique clipping region IDs
        self.clipping_stack: List[Optional[Dict[str, Any]]] = []  # Stack for nested clipping (with q/Q)
        # Track all active clip regions (nested clipping support)
        self.active_clip_stack: List[Dict[str, Any]] = []  # All currently active clips (outermost to innermost)
        self.clip_depth_stack: List[int] = []  # Stack of clip stack depths for q/Q restoration
        
        # Performance: Path parsing cache
        self._path_parse_cache: Dict[str, Any] = {}

        self.gstate = GraphicsState()
        self.gstack: List[GraphicsState] = []

        self.current_subpaths: List[List[Tuple]] = []
        self.current_subpath: List[Tuple] = []
        self.current_pt = (0.0, 0.0)

        self.vectors: List[Dict[str, Any]] = []

        self._patterns: Dict[str, Any] = {}
        self._shadings: Dict[str, Any] = {}
        self._extg: Dict[str, Any] = {}
        self._xobjects: Dict[str, Any] = {}
        self._color_spaces: Dict[str, Any] = {}

    # pikepdf deref
    @staticmethod
    def _deref(obj: Any) -> Any:
        """Dereference pikepdf indirect object."""
        try:
            return obj.get_object()
        except Exception:
            return obj

    def _is_image_xobject(self, xobj_name: str) -> bool:
        """Check if an XObject is an image (should be skipped in vector extraction)."""
        if xobj_name not in self._xobjects:
            return False
        
        try:
            xobj = self._deref(self._xobjects[xobj_name])
            subtype = xobj.get(KEY_SUBTYPE)
            if subtype:
                subtype_str = str(subtype).lstrip("/")
                return subtype_str == VAL_IMAGE.lstrip("/")
        except Exception as e:
            logger.debug(f"Error checking XObject type for {xobj_name}: {e}")
        
        return False

    def _load_resources(self, page) -> None:
        """Load PDF page resources (patterns, shadings, XObjects, etc.)."""
        self._patterns.clear()
        self._shadings.clear()
        self._extg.clear()
        self._xobjects.clear()
        self._color_spaces.clear()

        try:
            res = getattr(page, "Resources", None)
            if not res:
                return

            if KEY_EXT_GSTATE in res:
                for name, gs_obj in res.ExtGState.items():
                    self._extg[str(name).lstrip("/")] = gs_obj

            if KEY_PATTERN in res:
                for name, pat_obj in res.Pattern.items():
                    self._patterns[str(name).lstrip("/")] = pat_obj

            if KEY_SHADING in res:
                for name, sh_obj in res.Shading.items():
                    self._shadings[str(name).lstrip("/")] = sh_obj

            if KEY_XOBJECT in res:
                for name, xobj in res.XObject.items():
                    self._xobjects[str(name).lstrip("/")] = xobj
            
            if KEY_COLOR_SPACE in res:
                for name, cs_obj in res.ColorSpace.items():
                    self._color_spaces[str(name).lstrip("/")] = cs_obj
        except Exception as e:
            logger.debug(f"Resource load failed: {e}")

    def _handle_color_set(self, is_stroke: bool, inst, args: List[float]) -> None:
        """Handle SC/SCN/sc/scn color setting operators."""
        if not inst.operands:
            return

        last = inst.operands[-1]

        try:
            float(last)
            col = None
            if len(args) == 1:
                g = args[0]
                col = (g, g, g)
            elif len(args) == 3:
                col = (args[0], args[1], args[2])
            elif len(args) == 4:
                col = _cmyk_to_rgb(args[0], args[1], args[2], args[3])

            if col is not None:
                if is_stroke:
                    self.gstate.stroke_color = col
                    self.gstate.stroke_pattern = None
                else:
                    self.gstate.fill_color = col
                    self.gstate.fill_pattern = None
            return
        except Exception:
            pass

        name = str(last).lstrip("/")
        parsed = None

        if name in self._patterns:
            parsed = self._parse_pattern(self._patterns[name])
        elif name in self._shadings:
            parsed = self._parse_shading(self._shadings[name], pattern_matrix=None)

        if is_stroke:
            self.gstate.stroke_pattern = parsed
        else:
            self.gstate.fill_pattern = parsed

    def _parse_pattern(self, pat_obj) -> Optional[dict]:
        """
        Parse PDF pattern object.
        
        PDF Pattern Types:
        - Type 1 (Tiling): Repeating image/vector pattern -> returns {"type": "tiling", ...}
        - Type 2 (Shading): Gradient pattern -> delegates to _parse_shading() -> returns {"type": "linear"/"radial", ...}
        """
        pat_obj = self._deref(pat_obj)
        try:
            ptype = int(pat_obj.get(KEY_PATTERN_TYPE, 0))
        except Exception:
            ptype = 0
            
        pat_matrix = None
        try:
            m = pat_obj.get(KEY_MATRIX)
            if m and len(m) == 6:
                pat_matrix = [float(x) for x in m]
        except Exception:
            pat_matrix = None

        if ptype == 2:  # shading pattern (gradient)
            shading = pat_obj.get(KEY_SHADING)
            if shading is not None:
                return self._parse_shading(shading, pattern_matrix=pat_matrix)
        
        elif ptype == 1:  # tiling pattern
            # Extract images from Type 1 pattern resources
            try:
                if KEY_RESOURCES not in pat_obj:
                    return {
                        "type": "tiling",
                        "patternType": 1,
                        "images": [],
                        "matrix": pat_matrix
                    }
                
                resources = self._deref(pat_obj[KEY_RESOURCES])
                if KEY_XOBJECT not in resources:
                    return {
                        "type": "tiling",
                        "patternType": 1,
                        "images": [],
                        "matrix": pat_matrix
                    }
                
                xobjects = resources[KEY_XOBJECT]
                pattern_images = []
                
                for xobj_name, xobj in xobjects.items():
                    xobj_resolved = self._deref(xobj)
                    if xobj_resolved.get(KEY_SUBTYPE) == VAL_IMAGE:
                        # Extract image metadata
                        image_name = str(xobj_name).lstrip("/")
                        try:
                            width = int(xobj_resolved.get(KEY_WIDTH, 0))
                            height = int(xobj_resolved.get(KEY_HEIGHT, 0))
                            
                            # Extract image data if requested
                            image_data = None
                            if self.include_pattern_image_data:
                                try:
                                    from pikepdf import PdfImage
                                    pdf_image = PdfImage(xobj_resolved)
                                    pil_image = pdf_image.as_pil_image()
                                    
                                    # Convert to PNG and encode as base64
                                    img_buffer = io.BytesIO()
                                    pil_image.save(img_buffer, format='PNG')
                                    img_buffer.seek(0)
                                    image_data = base64.b64encode(img_buffer.read()).decode('utf-8')
                                except Exception as e:
                                    logger.warning(f"[PATTERN] Could not extract image data from pattern: {e}")
                            
                            pattern_images.append({
                                "name": image_name,
                                "width": width,
                                "height": height,
                                "data": image_data
                            })
                        except Exception as e:
                            logger.warning(f"[PATTERN] Error extracting image from pattern: {e}")
                
                if pattern_images:
                    logger.debug(f"[PATTERN] Tiling pattern with {len(pattern_images)} images")
                    
                return {
                    "type": "tiling",
                    "patternType": 1,
                    "images": pattern_images,
                    "matrix": pat_matrix
                }
            except Exception as e:
                logger.warning(f"[PATTERN] Error parsing Type 1 tiling pattern: {e}")
                return {
                    "type": "tiling",
                    "patternType": 1,
                    "images": [],
                    "matrix": pat_matrix
                }

        return None

    def _parse_shading(self, shading_obj, pattern_matrix: Optional[List[float]]) -> Optional[dict]:
        """Parse PDF shading object (gradients)."""
        shading_obj = self._deref(shading_obj)

        sh_matrix = None
        try:
            m = shading_obj.get(KEY_MATRIX)
            if m and len(m) == 6:
                sh_matrix = [float(x) for x in m]
        except Exception:
            sh_matrix = None

        # Combine matrices: CTM * Pattern Matrix * Shading Matrix
        combined = self.gstate.ctm[:]
        if pattern_matrix:
            combined = _mat_mul(combined, pattern_matrix)
        if sh_matrix:
            combined = _mat_mul(combined, sh_matrix)

        try:
            stype = int(shading_obj.get(KEY_SHADING_TYPE, 0))
        except Exception:
            stype = 0

        coords = []
        try:
            coords = [float(x) for x in shading_obj.get(KEY_COORDS, [])]
        except Exception:
            coords = []

        colors = self._extract_gradient_colors(shading_obj)

        if stype == 2 and len(coords) == 4:
            x1, y1, x2, y2 = coords
            if combined:
                x1, y1 = _mat_apply(combined, x1, y1)
                x2, y2 = _mat_apply(combined, x2, y2)
            return {"type": "linear", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "colors": colors}

        if stype == 3 and len(coords) == 6:
            x1, y1, r1, x2, y2, r2 = coords
            if combined:
                x1, y1 = _mat_apply(combined, x1, y1)
                x2, y2 = _mat_apply(combined, x2, y2)
            return {"type": "radial", "x1": x1, "y1": y1, "r1": r1, "x2": x2, "y2": y2, "r2": r2, "colors": colors}

        return None

    def _components_to_hex(self, comps) -> str:
        """Convert color components to hex string."""
        try:
            vals = [float(x) for x in comps]
        except Exception:
            vals = []

        # Handle different color spaces
        if len(vals) == 1:
            return _gray_to_hex(vals[0])
        elif len(vals) == 3:
            return _rgb_to_hex((vals[0], vals[1], vals[2]))
        elif len(vals) == 4:
            return _cmyk_to_hex(vals[0], vals[1], vals[2], vals[3])

        return "#808080"

    def _resolve_color_space(self, cs_name: str, components: List[float]) -> Optional[str]:
        """Resolve color from color space and components."""
        if not cs_name:
            return self._components_to_hex(components)
        
        # Device color spaces
        if cs_name in ("DeviceRGB", "RGB"):
            if len(components) == 3:
                return _rgb_to_hex(tuple(components))
        elif cs_name in ("DeviceGray", "Gray"):
            if len(components) == 1:
                return _gray_to_hex(components[0])
        elif cs_name in ("DeviceCMYK", "CMYK"):
            if len(components) == 4:
                return _cmyk_to_hex(*components)
        
        # ICCBased - treat as RGB-ish if N=3, CMYK-ish if N=4
        cs_obj = None
        if cs_name.startswith("/"):
            cs_obj = self._color_spaces.get(cs_name.lstrip("/"))
        
        if cs_obj:
            cs_obj = self._deref(cs_obj)
            if isinstance(cs_obj, list) and len(cs_obj) > 0:
                cs_type = str(cs_obj[0]).lstrip("/")
                
                if cs_type == "ICCBased":
                    # Get N parameter
                    icc_dict = self._deref(cs_obj[1]) if len(cs_obj) > 1 else None
                    if icc_dict:
                        n = int(icc_dict.get(KEY_N, 0))
                        if n == 3 and len(components) == 3:
                            return _rgb_to_hex(tuple(components))
                        elif n == 4 and len(components) == 4:
                            return _cmyk_to_hex(*components)
                        elif n == 1 and len(components) == 1:
                            return _gray_to_hex(components[0])
                
                elif cs_type == "Indexed":
                    # Indexed color space - lookup in palette
                    if len(components) == 1:
                        return self._lookup_indexed_color(cs_obj, int(components[0]))
        
        # Fallback
        return self._components_to_hex(components)

    def _lookup_indexed_color(self, cs_array, index: int) -> str:
        """Lookup color in indexed color space palette."""
        try:
            if len(cs_array) < 4:
                return "#808080"
            
            base_cs = cs_array[1]
            hival = int(cs_array[2])
            lookup_data = cs_array[3]
            
            if index < 0 or index > hival:
                return "#000000"
            
            # Get base color space component count
            base_cs_str = str(base_cs).lstrip("/")
            n_components = 3  # Default RGB
            if base_cs_str in ("DeviceGray", "Gray"):
                n_components = 1
            elif base_cs_str in ("DeviceCMYK", "CMYK"):
                n_components = 4
            
            # Extract color components from lookup table
            if hasattr(lookup_data, 'read_bytes'):
                data_bytes = lookup_data.read_bytes()
            else:
                data_bytes = bytes(lookup_data)
            
            offset = index * n_components
            if offset + n_components > len(data_bytes):
                return "#000000"
            
            components = [b / 255.0 for b in data_bytes[offset:offset + n_components]]
            
            # Convert to hex
            if n_components == 1:
                return _gray_to_hex(components[0])
            elif n_components == 3:
                return _rgb_to_hex(tuple(components))
            elif n_components == 4:
                return _cmyk_to_hex(*components)
        except Exception as e:
            logger.debug(f"Indexed color lookup failed: {e}")
        
        return "#808080"

    def _extract_gradient_colors(self, shading_obj) -> List[dict]:
        """Extract color stops from shading function."""
        shading_obj = self._deref(shading_obj)

        # Background fallback (important!)
        bg_hex = None
        try:
            bg = shading_obj.get(KEY_BACKGROUND)
            if bg:
                bg_hex = self._components_to_hex(bg)
        except Exception:
            bg_hex = None

        # Domain normalization (for stitching bounds)
        d0, d1 = 0.0, 1.0
        try:
            dom = shading_obj.get(KEY_DOMAIN, [0, 1])
            if dom and len(dom) >= 2:
                d0, d1 = float(dom[0]), float(dom[1])
        except Exception:
            d0, d1 = 0.0, 1.0

        def norm(t: float) -> float:
            try:
                if d1 == d0:
                    return 0.0
                return max(0.0, min(1.0, (float(t) - d0) / (d1 - d0)))
            except Exception:
                return 0.0

        function = shading_obj.get(KEY_FUNCTION)
        if function is None:
            return self._fallback_stops(bg_hex)

        function = self._deref(function)

        # Some PDFs store as array
        if isinstance(function, (list, pikepdf.Array)):
            try:
                function = self._deref(function[0])
            except Exception:
                return self._fallback_stops(bg_hex)

        try:
            ftype = int(function.get("/FunctionType", 0))
        except Exception:
            ftype = 0

        # FunctionType 2: C0->C1
        if ftype == 2:
            c0 = function.get(KEY_C0, [0])
            c1 = function.get(KEY_C1, [1])
            return [
                {"offset": 0.0, "color": self._components_to_hex(c0)},
                {"offset": 0.5, "color": self._components_to_hex(self._mix(c0, c1, 0.5))},
                {"offset": 1.0, "color": self._components_to_hex(c1)},
            ]

        # FunctionType 3: stitching
        if ftype == 3:
            try:
                subs = [self._deref(f) for f in function.get(KEY_FUNCTIONS, [])]
                bounds = [float(b) for b in function.get(KEY_BOUNDS, [])]
            except Exception:
                return self._fallback_stops(bg_hex)

            stops: List[dict] = []
            if subs:
                stops.append({"offset": 0.0, "color": self._components_to_hex(subs[0].get(KEY_C0, [0]))})

            for i, b in enumerate(bounds):
                if i < len(subs):
                    stops.append({"offset": norm(b), "color": self._components_to_hex(subs[i].get(KEY_C1, [1]))})

            if subs:
                stops.append({"offset": 1.0, "color": self._components_to_hex(subs[-1].get(KEY_C1, [1]))})

            stops.sort(key=lambda s: s["offset"])
            # compact duplicates
            out = []
            for s in stops:
                if not out or out[-1]["color"] != s["color"] or abs(out[-1]["offset"] - s["offset"]) > 1e-6:
                    out.append(s)
            return out or self._fallback_stops(bg_hex)

        # FunctionType 0 (sampled) and 4 (PostScript) should fallback
        return self._fallback_stops(bg_hex)

    @staticmethod
    def _mix(a, b, t: float) -> List[float]:
        """Mix two color arrays with interpolation factor t."""
        try:
            aa = [float(x) for x in a]
            bb = [float(x) for x in b]
            n = min(len(aa), len(bb))
            return [(aa[i] * (1 - t) + bb[i] * t) for i in range(n)]
        except Exception:
            return [0.5]

    @staticmethod
    def _is_gradient_pattern(pattern: Optional[dict]) -> bool:
        """Check if pattern is a gradient (linear or radial shading)."""
        if not pattern or not isinstance(pattern, dict):
            return False
        pattern_type = pattern.get("type")
        return pattern_type in ("linear", "radial")
    
    @staticmethod
    def _is_tiling_pattern(pattern: Optional[dict]) -> bool:
        """Check if pattern is a tiling pattern (Type 1 repeating pattern)."""
        if not pattern or not isinstance(pattern, dict):
            return False
        return pattern.get("type") == "tiling"
    
    @staticmethod
    def _get_pattern_type_name(pattern: Optional[dict]) -> str:
        """Get human-readable pattern type name for logging/debugging."""
        if not pattern:
            return "none"
        pattern_type = pattern.get("type", "unknown")
        if pattern_type == "linear":
            return "linear gradient"
        elif pattern_type == "radial":
            return "radial gradient"
        elif pattern_type == "tiling":
            return "tiling pattern"
        return f"unknown ({pattern_type})"

    @staticmethod
    def _fallback_stops(bg_hex: Optional[str]) -> List[dict]:
        """Fallback gradient color stops."""
        if bg_hex:
            return [{"offset": 0.0, "color": bg_hex}, {"offset": 1.0, "color": bg_hex}]
        return [{"offset": 0.0, "color": "#808080"}, {"offset": 1.0, "color": "#c0c0c0"}]

    def _svg_pattern(self, pattern_id: str, tiling_pattern: dict, offset_x: float, offset_y: float) -> str:
        """Generate SVG pattern element for tiling patterns."""
        if not tiling_pattern or tiling_pattern.get("type") != "tiling":
            logger.warning(f"[PATTERN] _svg_pattern called with non-tiling pattern: {tiling_pattern.get('type')}")
            return ""
        
        images = tiling_pattern.get("images", [])
        if not images:
            logger.warning(f"[PATTERN] Tiling pattern has no images")
            return ""
        
        # Get pattern matrix for sizing (if available)
        matrix = tiling_pattern.get("matrix")
        
        # Use first image for pattern (could be enhanced to composite multiple images)
        first_image = images[0]
        width = first_image.get("width", 100)
        height = first_image.get("height", 100)
        
        # Build pattern element
        pattern_xml = (
            f'<pattern id="{pattern_id}" x="0" y="0" width="{width}" height="{height}" '
            f'patternUnits="userSpaceOnUse">'
        )
        
        # Add images to pattern
        for img in images:
            img_width = img.get("width", 100)
            img_height = img.get("height", 100)
            img_data = img.get("data")
            
            if img_data:
                # Image data is base64 encoded
                pattern_xml += (
                    f'<image x="0" y="0" width="{img_width}" height="{img_height}" '
                    f'href="data:image/png;base64,{img_data}" />'
                )
            else:
                # No image data available, use a placeholder rect
                pattern_xml += (
                    f'<rect x="0" y="0" width="{img_width}" height="{img_height}" '
                    f'fill="#cccccc" />'
                )
        
        pattern_xml += '</pattern>'
        
        return pattern_xml

    # --- svg gradient defs ---
    def _svg_gradient(self, grad_id: str, pattern: dict, offset_x: float, offset_y: float) -> str:
        """Generate SVG gradient definition."""
        if not pattern:
            logger.warning(f"[GRADIENT] _svg_gradient called with empty pattern for ID: {grad_id}")
            return ""
        
        def stops_xml(stops: List[dict]) -> str:
            parts = []
            for s in stops:
                off = max(0.0, min(1.0, float(s.get("offset", 0.0))))
                col = s.get("color", "#808080")
                parts.append(f'<stop offset="{off:.6f}" stop-color="{col}" />')
            return "".join(parts)

        if pattern["type"] == "linear":
            x1 = pattern["x1"] - offset_x
            y1 = self.page_height - (pattern["y1"] - self.mediabox_bottom) - offset_y
            x2 = pattern["x2"] - offset_x
            y2 = self.page_height - (pattern["y2"] - self.mediabox_bottom) - offset_y
            return (
                f'<linearGradient id="{grad_id}" x1="{x1:.3f}" y1="{y1:.3f}" '
                f'x2="{x2:.3f}" y2="{y2:.3f}" gradientUnits="userSpaceOnUse">'
                f"{stops_xml(pattern.get('colors', []))}</linearGradient>"
            )

        if pattern["type"] == "radial":
            cx = pattern["x2"] - offset_x
            cy = self.page_height - (pattern["y2"] - self.mediabox_bottom) - offset_y
            r = float(pattern.get("r2", 0.0))
            fx = pattern["x1"] - offset_x
            fy = self.page_height - (pattern["y1"] - self.mediabox_bottom) - offset_y
            return (
                f'<radialGradient id="{grad_id}" cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
                f'fx="{fx:.3f}" fy="{fy:.3f}" gradientUnits="userSpaceOnUse">'
                f"{stops_xml(pattern.get('colors', []))}</radialGradient>"
            )

        logger.warning(f"[GRADIENT] Unknown pattern type '{pattern.get('type')}' for gradient ID: {grad_id}")
        return ""

    # --- path handling ---
    def _reset_path(self) -> None:
        """Reset path tracking state."""
        self.current_subpaths = []
        self.current_subpath = []
        self.current_pt = (0.0, 0.0)

    def _parse_path_segment(self, op: str, args: List[float]) -> None:
        """Parse path construction operators (m, l, c, v, y, re, h)."""
        if op == "m" and len(args) >= 2:
            if self.current_subpath:
                self.current_subpaths.append(self.current_subpath)
            self.current_subpath = [("m", args[0], args[1])]
            self.current_pt = (args[0], args[1])

        elif op == "l" and len(args) >= 2:
            self.current_subpath.append(("l", args[0], args[1]))
            self.current_pt = (args[0], args[1])

        elif op == "c" and len(args) >= 6:
            self.current_subpath.append(("c", *args[:6]))
            self.current_pt = (args[4], args[5])

        elif op == "v" and len(args) >= 4:
            self.current_subpath.append(("c", self.current_pt[0], self.current_pt[1], args[0], args[1], args[2], args[3]))
            self.current_pt = (args[2], args[3])

        elif op == "y" and len(args) >= 4:
            self.current_subpath.append(("c", args[0], args[1], args[2], args[3], args[2], args[3]))
            self.current_pt = (args[2], args[3])

        elif op == "re" and len(args) >= 4:
            self.current_subpath.append(("re", *args[:4]))

        elif op == "h":
            self.current_subpath.append(("h",))

    def _build_paths_and_bbox(self, subpaths: List[List[Tuple]], has_stroke: bool) -> Tuple[List[str], Optional[Tuple[float, float, float, float]]]:
        """Build SVG paths and calculate bounding box."""
        svg_paths: List[str] = []
        pts: List[Tuple[float, float]] = []

        for sub in subpaths:
            cmds = []
            for seg in sub:
                sop = seg[0]
                if sop in ("m", "l"):
                    x, y = self.gstate.transform_point(seg[1], seg[2])
                    # Adjust for MediaBox bottom offset, then flip Y
                    y = self.page_height - (y - self.mediabox_bottom)
                    cmds.append(f"{sop.upper()} {x:.3f} {y:.3f}")
                    pts.append((x, y))

                elif sop == "c":
                    p1 = self.gstate.transform_point(seg[1], seg[2])
                    p2 = self.gstate.transform_point(seg[3], seg[4])
                    p3 = self.gstate.transform_point(seg[5], seg[6])
                    # Adjust for MediaBox bottom offset, then flip Y
                    p1 = (p1[0], self.page_height - (p1[1] - self.mediabox_bottom))
                    p2 = (p2[0], self.page_height - (p2[1] - self.mediabox_bottom))
                    p3 = (p3[0], self.page_height - (p3[1] - self.mediabox_bottom))
                    cmds.append(
                        f"C {p1[0]:.3f} {p1[1]:.3f}, {p2[0]:.3f} {p2[1]:.3f}, {p3[0]:.3f} {p3[1]:.3f}"
                    )
                    pts.extend([p1, p2, p3])

                elif sop == "re":
                    rx, ry, rw, rh = seg[1:]
                    p0 = self.gstate.transform_point(rx, ry)
                    p1 = self.gstate.transform_point(rx + rw, ry + rh)
                    # Adjust for MediaBox bottom offset, then flip Y
                    p0 = (p0[0], self.page_height - (p0[1] - self.mediabox_bottom))
                    p1 = (p1[0], self.page_height - (p1[1] - self.mediabox_bottom))
                    cmds.append(f"M {p0[0]:.3f} {p0[1]:.3f} H {p1[0]:.3f} V {p1[1]:.3f} H {p0[0]:.3f} Z")
                    pts.extend([p0, p1])

                elif sop == "h":
                    cmds.append("Z")

            d = " ".join(cmds).strip()
            if d:
                svg_paths.append(d)

        if not pts:
            return [], None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = self.gstate.line_width if has_stroke else 0.0
        return svg_paths, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def _build_svg(self, paths_abs: List[str], x: float, y: float, w: float, h: float, has_stroke: bool, has_fill: bool, fill_rule: str, clip_stack: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Build complete SVG markup with optimized viewBox.
        """
        # Combine into a single compound path (holes)
        combined = []
        for d in paths_abs:
            parts = []
            for m in _CMD_RE.finditer(d):
                cmd = m.group(1)
                coords_str = m.group(2).strip()

                if cmd == "Z":
                    parts.append("Z")
                elif cmd == "H":
                    if coords_str:
                        xv = float(coords_str) - x
                        parts.append(f"H {xv:.3f}")
                elif cmd == "V":
                    if coords_str:
                        yv = float(coords_str) - y
                        parts.append(f"V {yv:.3f}")
                elif coords_str:
                    nums = _NUM_RE.findall(coords_str)
                    shifted = [float(nums[i]) - (x if i % 2 == 0 else y) for i in range(len(nums))]
                    parts.append(f"{cmd} " + " ".join(f"{n:.3f}" for n in shifted))

            combined.append(" ".join(parts))

        combined_d = re.sub(r"\s+", " ", " ".join(combined)).strip()

        gradient_defs: List[str] = []

        # fill
        if has_fill:
            if self.gstate.fill_pattern:
                pattern_type_name = self._get_pattern_type_name(self.gstate.fill_pattern)
                
                # Only create SVG gradients for actual gradients (linear/radial), not tiling patterns
                if self._is_gradient_pattern(self.gstate.fill_pattern):
                    gid = f"fill_grad_{abs(hash(str(self.gstate.fill_pattern)))}"
                    grad_def = self._svg_gradient(gid, self.gstate.fill_pattern, x, y)
                    
                    if not grad_def:
                        logger.warning(f"[GRADIENT] Empty gradient definition for {pattern_type_name}")
                    else:
                        gradient_defs.append(grad_def)
                        fill_ref = f"url(#{gid})"
                elif self._is_tiling_pattern(self.gstate.fill_pattern):
                    # Tiling patterns: generate SVG <pattern> element
                    pid = f"fill_pattern_{abs(hash(str(self.gstate.fill_pattern)))}"
                    pattern_def = self._svg_pattern(pid, self.gstate.fill_pattern, x, y)
                    
                    if pattern_def:
                        gradient_defs.append(pattern_def)  # Reuse gradient_defs list for all defs
                        fill_ref = f"url(#{pid})"
                    else:
                        # Fallback if pattern generation fails
                        logger.warning(f"[PATTERN] Failed to generate SVG pattern, using fallback color")
                        fill_ref = "#cccccc"
                else:
                    # Unknown pattern type
                    logger.warning(f"[PATTERN] Unknown fill pattern type: {pattern_type_name}, using fallback color")
                    fill_ref = "#cccccc"  # Gray fallback for unknown patterns
            else:
                fill_ref = _rgb_to_hex(self.gstate.fill_color)
        else:
            fill_ref = "none"

        # stroke
        if has_stroke:
            if self.gstate.stroke_pattern:
                pattern_type_name = self._get_pattern_type_name(self.gstate.stroke_pattern)
                
                # Only create SVG gradients for actual gradients (linear/radial), not tiling patterns
                if self._is_gradient_pattern(self.gstate.stroke_pattern):
                    gid = f"stroke_grad_{abs(hash(str(self.gstate.stroke_pattern)))}"
                    grad_def = self._svg_gradient(gid, self.gstate.stroke_pattern, x, y)
                    
                    if grad_def:
                        gradient_defs.append(grad_def)
                        stroke_ref = f"url(#{gid})"
                elif self._is_tiling_pattern(self.gstate.stroke_pattern):
                    # Tiling patterns: generate SVG <pattern> element
                    pid = f"stroke_pattern_{abs(hash(str(self.gstate.stroke_pattern)))}"
                    pattern_def = self._svg_pattern(pid, self.gstate.stroke_pattern, x, y)
                    
                    if pattern_def:
                        gradient_defs.append(pattern_def)
                        stroke_ref = f"url(#{pid})"
                    else:
                        logger.warning(f"[PATTERN] Failed to generate SVG pattern, using fallback color")
                        stroke_ref = "#333333"
                else:
                    logger.warning(f"[PATTERN] Unknown stroke pattern type: {pattern_type_name}, using fallback color")
                    stroke_ref = "#333333"
            else:
                stroke_ref = _rgb_to_hex(self.gstate.stroke_color)
        else:
            stroke_ref = "none"

        style = [
            f'stroke="{stroke_ref}"',
            f'stroke-width="{self.gstate.line_width}"',
            f'fill="{fill_ref}"',
            f'fill-rule="{fill_rule}"',
        ]

        if has_fill and self.gstate.fill_opacity < 0.999:
            style.append(f'fill-opacity="{self.gstate.fill_opacity:.3f}"')
        if has_stroke and self.gstate.stroke_opacity < 0.999:
            style.append(f'stroke-opacity="{self.gstate.stroke_opacity:.3f}"')

        if has_stroke:
            # Line cap: 0=butt, 1=round, 2=square
            cap_names = ["butt", "round", "square"]
            if 0 <= self.gstate.line_cap < len(cap_names):
                style.append(f'stroke-linecap="{cap_names[self.gstate.line_cap]}"')
            # Line join: 0=miter, 1=round, 2=bevel
            join_names = ["miter", "round", "bevel"]
            if 0 <= self.gstate.line_join < len(join_names):
                style.append(f'stroke-linejoin="{join_names[self.gstate.line_join]}"')
            if self.gstate.miter_limit != 10.0:
                style.append(f'stroke-miterlimit="{self.gstate.miter_limit}"')

        if has_stroke and self.gstate.dash_array:
            style.append(f'stroke-dasharray="{" ".join(str(d) for d in self.gstate.dash_array)}"')
            if self.gstate.dash_phase:
                style.append(f'stroke-dashoffset="{self.gstate.dash_phase}"')

        defs = ""
        gd = [g for g in gradient_defs if g]
        
        # Generate clipPath definitions if we have active clips
        clip_defs = []
        if clip_stack:
            for clip_region in clip_stack:
                clip_id = clip_region.get("id")
                clip_path_d = clip_region.get("path")
                clip_rule = clip_region.get("rule", "nonzero")
                
                if clip_path_d:
                    # Shift clip path coordinates relative to viewBox origin (x, y)
                    shifted_clip_path = self._shift_path_coords(clip_path_d, x, y)
                    # Ensure clip path is closed (add Z if not present)
                    if shifted_clip_path and not shifted_clip_path.rstrip().endswith('Z'):
                        shifted_clip_path = shifted_clip_path + ' Z'
                    clip_defs.append(
                        f'<clipPath id="clip_{clip_id}">'
                        f'<path d="{shifted_clip_path}" clip-rule="{clip_rule}" />'
                        f'</clipPath>'
                    )
        
        all_defs = gd + clip_defs
        if all_defs:
            defs = "<defs>" + "".join(all_defs) + "</defs>"

        # Build the path element
        path_elem = f'<path d="{combined_d}" {" ".join(style)} />'
        
        # Wrap in nested <g> elements with clip-path if we have clips
        if clip_stack:
            for clip_region in reversed(clip_stack):
                clip_id = clip_region.get("id")
                path_elem = f'<g clip-path="url(#clip_{clip_id})">{path_elem}</g>'

        return (
            f'<svg viewBox="0 0 {w:.2f} {h:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'xmlns="http://www.w3.org/2000/svg">{defs}'
            f'{path_elem}</svg>'
        )

    def _shift_path_coords(self, path_d: str, offset_x: float, offset_y: float) -> str:
        """
        Shift all coordinates in an SVG path 'd' attribute by the given offsets.
        Used to transform clip paths relative to the viewBox origin.
        """
        if not path_d:
            return ""
        
        parts = []
        for m in _CMD_RE.finditer(path_d):
            cmd = m.group(1)
            coords_str = m.group(2).strip()
            
            if cmd == "Z":
                parts.append("Z")
            elif cmd == "H":
                if coords_str:
                    xv = float(coords_str) - offset_x
                    parts.append(f"H {xv:.3f}")
            elif cmd == "V":
                if coords_str:
                    yv = float(coords_str) - offset_y
                    parts.append(f"V {yv:.3f}")
            elif coords_str:
                nums = _NUM_RE.findall(coords_str)
                shifted = [float(nums[i]) - (offset_x if i % 2 == 0 else offset_y) for i in range(len(nums))]
                parts.append(f"{cmd} " + " ".join(f"{n:.3f}" for n in shifted))
        
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _emit_shading_rect(self, x: float, y: float, w: float, h: float, shading_info: dict) -> None:
        """Emit a rectangle filled with shading gradient."""
        # Transform coordinates to page space
        x1, y1 = self.gstate.transform_point(x, y)
        x2, y2 = self.gstate.transform_point(x + w, y + h)
        
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        w_abs, h_abs = max_x - min_x, max_y - min_y
        
        if w_abs < 0.1 or h_abs < 0.1:
            return
        
        # Build SVG with gradient
        gid = f"sh_grad_{abs(hash(str(shading_info)))}"
        grad_def = self._svg_gradient(gid, shading_info, min_x, min_y)
        
        # Rectangle path
        rect_path = f"M 0 0 L {w_abs:.2f} 0 L {w_abs:.2f} {h_abs:.2f} L 0 {h_abs:.2f} Z"
        
        svg = (
            f'<svg viewBox="0 0 {w_abs:.2f} {h_abs:.2f}" width="{w_abs:.2f}" height="{h_abs:.2f}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<defs>{grad_def}</defs>'
            f'<path d="{rect_path}" fill="url(#{gid})" />'
            f'</svg>'
        )
        
        self.vectors.append({
            "x": min_x,
            "y": min_y,
            "width": w_abs,
            "height": h_abs,
            "svgContent": svg,
            "dataUri": f"data:image/svg+xml;base64,{_b64(svg)}",
            "stroke": None,
            "fill": "gradient",
            "strokeWidth": None,
            "pathCount": 1,
            "needsRasterize": False,
            "hasSoftMask": self.gstate.has_soft_mask,
        })

    def _process_form_xobject(self, xobj, xobj_name: Optional[str] = None) -> None:
        """
        Process Form XObject content stream to extract vectors inside it.
        Form XObjects often contain opacity-affected graphics and backgrounds.
        
        Args:
            xobj: The Form XObject to process
            xobj_name: Name of the Form XObject (e.g., "TG2", "TG4")
        """
        # Save current graphics state (including opacity from parent)
        saved_state = self.gstate.copy()
        parent_fill_opacity = self.gstate.fill_opacity
        parent_stroke_opacity = self.gstate.stroke_opacity
        
        # Apply Form XObject's transformation matrix
        form_matrix = xobj.get(KEY_MATRIX, [1, 0, 0, 1, 0, 0])
        if form_matrix and len(form_matrix) == 6:
            self.gstate.apply_ctm(*[float(x) for x in form_matrix])
        
        # Check for transparency group and extract opacity
        group = xobj.get(VAL_GROUP)
        if group:
            group = self._deref(group)
            # Note: Transparency group opacity is already in graphics state from /gs
        
        # Load Form XObject's resources
        form_resources = xobj.get(KEY_RESOURCES)
        if form_resources:
            form_resources = self._deref(form_resources)
            # Temporarily update resources for this form
            saved_patterns = self._patterns.copy()
            saved_shadings = self._shadings.copy()
            saved_extg = self._extg.copy()
            
            try:
                if KEY_PATTERN in form_resources:
                    for pname, pat_obj in form_resources.Pattern.items():
                        self._patterns[str(pname).lstrip("/")] = pat_obj
                if KEY_SHADING in form_resources:
                    for sname, sh_obj in form_resources.Shading.items():
                        self._shadings[str(sname).lstrip("/")] = sh_obj
                if KEY_EXT_GSTATE in form_resources:
                    for gname, gs_obj in form_resources.ExtGState.items():
                        self._extg[str(gname).lstrip("/")] = gs_obj
            except Exception:
                pass
        
        # Process Form XObject's content stream
        try:
            form_ops = parse_content_stream(xobj)
            
            # Process each operator in the form
            for form_inst in form_ops:
                form_op = str(form_inst.operator)
                
                # Extract numeric args
                form_args: List[float] = []
                for x in form_inst.operands:
                    try:
                        form_args.append(float(x))
                    except Exception:
                        pass
                
                # Handle graphics state
                if form_op == _op_str(OP_SAVE_STATE):
                    self.gstack.append(self.gstate.copy())
                elif form_op == _op_str(OP_RESTORE_STATE):
                    if self.gstack:
                        self.gstate = self.gstack.pop()
                elif form_op == _op_str(OP_CTM) and len(form_args) == 6:
                    self.gstate.apply_ctm(*form_args)
                
                # Handle extended graphics state (opacity)
                elif form_op == _op_str(OP_SET_GRAPHICS_STATE_PARAMS) and form_inst.operands:
                    gs_name = str(form_inst.operands[0]).lstrip("/")
                    gs = self._extg.get(gs_name)
                    if gs is not None:
                        gs = self._deref(gs)
                        try:
                            # Multiply form opacity with parent opacity (PDF spec behavior)
                            if KEY_FILL_OPACITY in gs:
                                form_opacity = float(gs[KEY_FILL_OPACITY])
                                self.gstate.fill_opacity = parent_fill_opacity * form_opacity
                            if KEY_STROKE_OPACITY in gs:
                                form_opacity = float(gs[KEY_STROKE_OPACITY])
                                self.gstate.stroke_opacity = parent_stroke_opacity * form_opacity
                        except Exception:
                            pass
                
                # Handle colors
                elif form_op == _op_str(OP_SET_RGB_COLOR_STROKE) and len(form_args) == 3:
                    self.gstate.stroke_color = (form_args[0], form_args[1], form_args[2])
                    self.gstate.stroke_pattern = None
                elif form_op == _op_str(OP_SET_RGB_COLOR_FILL) and len(form_args) == 3:
                    self.gstate.fill_color = (form_args[0], form_args[1], form_args[2])
                    self.gstate.fill_pattern = None
                elif form_op == _op_str(OP_SET_GRAY_STROKE) and len(form_args) == 1:
                    g = form_args[0]
                    self.gstate.stroke_color = (g, g, g)
                    self.gstate.stroke_pattern = None
                elif form_op == _op_str(OP_SET_GRAY_FILL) and len(form_args) == 1:
                    g = form_args[0]
                    self.gstate.fill_color = (g, g, g)
                    self.gstate.fill_pattern = None
                elif form_op == _op_str(OP_SET_CMYK_COLOR_STROKE) and len(form_args) == 4:
                    self.gstate.stroke_color = _cmyk_to_rgb(form_args[0], form_args[1], form_args[2], form_args[3])
                    self.gstate.stroke_pattern = None
                elif form_op == _op_str(OP_SET_CMYK_COLOR_FILL) and len(form_args) == 4:
                    self.gstate.fill_color = _cmyk_to_rgb(form_args[0], form_args[1], form_args[2], form_args[3])
                    self.gstate.fill_pattern = None
                elif form_op in (_op_str(OP_SET_COLOR_STROKE), _op_str(OP_SET_COLOR_STROKE_N)):
                    self._handle_color_set(is_stroke=True, inst=form_inst, args=form_args)
                elif form_op in (_op_str(OP_SET_COLOR_FILL), _op_str(OP_SET_COLOR_FILL_N)):
                    self._handle_color_set(is_stroke=False, inst=form_inst, args=form_args)
                
                # Handle path construction
                elif form_op in _PATH_OPS_STR:
                    self._parse_path_segment(form_op, form_args)
                
                # Handle path painting (emit vectors)
                elif form_op in _PAINT_OPS_STR:
                    if form_op != _END_PATH_STR:
                        has_stroke = form_op in _STROKE_OPS_STR
                        has_fill = form_op in _FILL_OPS_STR
                        fill_rule = "evenodd" if form_op in _EVEN_ODD_OPS_STR else "nonzero"
                        self._emit_vector(has_stroke, has_fill, fill_rule)
                    self._reset_path()
                
                # Handle shadings
                elif form_op == _op_str(OP_SHADING) and form_inst.operands:
                    sh_name = str(form_inst.operands[0]).lstrip("/")
                    sh = self._shadings.get(sh_name)
                    if sh is not None:
                        self.gstate.fill_pattern = self._parse_shading(sh, pattern_matrix=None)
                        if self.current_subpath or self.current_subpaths:
                            self._emit_vector(has_stroke=False, has_fill=True, fill_rule="nonzero")
                            self._reset_path()
        
        except Exception as e:
            logger.debug(f"Error processing Form XObject content: {e}")
        
        # Restore resources if they were saved
        if form_resources:
            try:
                self._patterns = saved_patterns
                self._shadings = saved_shadings
                self._extg = saved_extg
            except Exception:
                pass
        
        # Restore graphics state
        self.gstate = saved_state
    
    def _emit_vector(self, has_stroke: bool, has_fill: bool, fill_rule: str) -> None:
        """Emit a vector graphics element."""
        if self.current_subpath:
            self.current_subpaths.append(self.current_subpath)
        if not self.current_subpaths:
            return

        svg_paths_abs, bbox = self._build_paths_and_bbox(self.current_subpaths, has_stroke)
        if not svg_paths_abs or bbox is None:
            return

        min_x, min_y, max_x, max_y = bbox
        w, h = max_x - min_x, max_y - min_y
        if w < 0.1 or h < 0.1:
            return
        
        # Determine which clip stack to pass for SVG generation
        clip_stack_for_svg = self.active_clip_stack if self.active_clip_stack else None

        # Only generate SVG if requested (expensive operation)
        svg = self._build_svg(svg_paths_abs, min_x, min_y, w, h, has_stroke, has_fill, fill_rule, clip_stack_for_svg) if self.generate_svg else ""

        # Determine if rasterization fallback is needed
        has_gradient = bool(self.gstate.fill_pattern or self.gstate.stroke_pattern)
        needs_blend_raster = self.gstate.blend_mode and self.gstate.blend_mode != "Normal"
        needs_rasterize = (
            self.gstate.has_soft_mask or
            needs_blend_raster
        )

        # Determine fill value based on pattern type
        fill_value = None
        if has_fill:
            if self.gstate.fill_pattern:
                pattern_type = self.gstate.fill_pattern.get("type") if isinstance(self.gstate.fill_pattern, dict) else None
                if pattern_type == "tiling":
                    fill_value = "pattern"
                elif pattern_type in ("linear", "radial"):
                    fill_value = "gradient"
                else:
                    fill_value = "unknown_pattern"
            else:
                fill_value = _rgb_to_hex(self.gstate.fill_color)
        
        # Determine stroke value based on pattern type
        stroke_value = None
        if has_stroke:
            if self.gstate.stroke_pattern:
                pattern_type = self.gstate.stroke_pattern.get("type") if isinstance(self.gstate.stroke_pattern, dict) else None
                if pattern_type == "tiling":
                    stroke_value = "pattern"
                elif pattern_type in ("linear", "radial"):
                    stroke_value = "gradient"
                else:
                    stroke_value = "unknown_pattern"
            else:
                stroke_value = _rgb_to_hex(self.gstate.stroke_color)
        
        # Build base vector element
        vector_element = {
            "x": min_x,
            "y": min_y,
            "width": w,
            "height": h,
            "svgContent": svg,
            "dataUri": f"data:image/svg+xml;base64,{_b64(svg)}",
            "stroke": stroke_value,
            "fill": fill_value,
            "strokeWidth": self.gstate.line_width if has_stroke else None,
            "pathCount": len(self.current_subpaths),
            "needsRasterize": needs_rasterize,
            "hasSoftMask": self.gstate.has_soft_mask,
            # Clipping region grouping metadata (skip for circular sun TEST)
            "clippingRegionId": self.active_clipping_region["id"] if self.active_clipping_region else None,
            "clippingPath": self.active_clipping_region.get("path") if self.active_clipping_region else None,
            "clippingRule": self.active_clipping_region.get("rule") if self.active_clipping_region else None,
            # Store vector's own bounding box for clipPath intersection
            "vectorBounds": {"x": min_x, "y": min_y, "width": w, "height": h},
            # Stream order tracking for z-order preservation
            "stream_index": len(self.vectors),
        }

        # Add pattern metadata if fill uses a tiling pattern (Type 1) with images
        if has_fill and self.gstate.fill_pattern:
            pattern_info = self.gstate.fill_pattern
            if isinstance(pattern_info, dict) and pattern_info.get("type") == "tiling":
                vector_element["fillType"] = "pattern"
                vector_element["patternType"] = pattern_info.get("patternType")
                
                # Include all images from pattern
                images = pattern_info.get("images", [])
                if images:
                    vector_element["patternImages"] = images
            elif pattern_info and pattern_info.get("type") in ("linear", "radial"):  # Gradient pattern (Type 2)
                vector_element["fillType"] = "gradient"
        elif has_fill:
            vector_element["fillType"] = "solid"

        # Stream-based sparse vector splitting (if enabled)
        if self.split_sparse_vectors:
            self._emit_with_splitting(vector_element, svg_paths_abs, min_x, min_y, w, h, has_stroke, has_fill, fill_rule)
        else:
            # Stream-based overlap merging (if enabled)
            if self.enable_overlap_grouping:
                merged = self._check_and_merge_overlap(vector_element)
                if not merged:
                    # Only append if not merged with existing vectors
                    self.vectors.append(vector_element)
            else:
                self.vectors.append(vector_element)
    
    def _check_and_merge_overlap(self, vector_element: dict) -> bool:
        """
        Check if new vector overlaps with recent vectors and merge if needed.
        """
        from utils.vector_grouping import find_overlapping_vectors, merge_vectors
        
        # Get recent vectors to check (last N vectors)
        start_idx = max(0, len(self.vectors) - self.overlap_check_window)
        recent_vectors = self.vectors[start_idx:]
        
        if not recent_vectors:
            return False
        
        # Find overlapping vectors
        overlapping_indices = find_overlapping_vectors(
            vector_element,
            recent_vectors,
            overlap_method=self.overlap_method,
            overlap_threshold=self.overlap_threshold,
            proximity_threshold=self.proximity_threshold
        )
        
        if overlapping_indices:
            # Get the actual indices in self.vectors
            actual_indices = [start_idx + i for i in overlapping_indices]
            overlapping_vectors = [self.vectors[i] for i in actual_indices]
            
            # Merge all overlapping vectors with the new one
            merged_vector = merge_vectors([vector_element] + overlapping_vectors)
            
            # Remove the old overlapping vectors (in reverse to maintain indices)
            for idx in sorted(actual_indices, reverse=True):
                del self.vectors[idx]
            
            # Append the merged vector
            self.vectors.append(merged_vector)
            
            logger.debug(
                f"[OVERLAP MERGE] Merged {len(overlapping_vectors)+1} vectors "
                f"into one (union bounds: {merged_vector['width']:.1f}x{merged_vector['height']:.1f})"
            )
            
            return True  # Vector was merged, don't append original
        
        return False  # No overlap, append normally
    
    def _emit_with_splitting(self, vector_element: dict, svg_paths_abs: List[str], min_x: float, min_y: float, w: float, h: float, has_stroke: bool, has_fill: bool, fill_rule: str) -> None:
        """
        Emit vector with stream-based splitting if it's sparse.
        Similar to how text runs are grouped - process on detection.
        """
        from utils.content_bounds import calculate_path_segments_bounds, merge_nearby_bounds, extract_subpath_from_svg
        
        vector_area = w * h
        coverage_percent = (vector_area / self.page_area) * 100
        
        # Check if vector is large enough to warrant analysis
        if coverage_percent < self.sparse_coverage_threshold:
            self.vectors.append(vector_element)
            return
        
        # Analyze path segments for splitting
        svg = vector_element.get("svgContent", "")
        if not svg:
            self.vectors.append(vector_element)
            return
        
        import re
        path_match = re.search(r'd="([^"]+)"', svg)
        if not path_match:
            self.vectors.append(vector_element)
            return
        
        path_d = path_match.group(1)
        
        try:
            # Calculate segment bounds
            segments = calculate_path_segments_bounds(path_d)
            
            if len(segments) <= 1:
                self.vectors.append(vector_element)
                return
            
            # Merge nearby segments with configured distance
            merged = merge_nearby_bounds(segments, self.sparse_merge_distance)
            
            if len(merged) <= 1:
                self.vectors.append(vector_element)
                return
            
            # Split into multiple vectors on stream
            logger.debug(f"[STREAM SPLIT] Splitting vector (coverage={coverage_percent:.1f}%) into {len(merged)} sub-vectors")
            
            for sub_idx, (seg_min_x, seg_min_y, seg_max_x, seg_max_y) in enumerate(merged):
                seg_w = seg_max_x - seg_min_x
                seg_h = seg_max_y - seg_min_y
                
                if seg_w < 0.1 or seg_h < 0.1:
                    continue
                
                # Extract sub-SVG for this bound
                sub_svg = extract_subpath_from_svg(svg, (seg_min_x, seg_min_y, seg_max_x, seg_max_y))
                
                if not sub_svg:
                    continue
                
                # Create sub-vector element
                sub_vector = {
                    "x": seg_min_x,
                    "y": seg_min_y,
                    "width": seg_w,
                    "height": seg_h,
                    "svgContent": sub_svg,
                    "dataUri": f"data:image/svg+xml;base64,{_b64(sub_svg)}",
                    "stroke": vector_element.get("stroke"),
                    "fill": vector_element.get("fill"),
                    "strokeWidth": vector_element.get("strokeWidth"),
                    "pathCount": 1,  # Each split is typically one segment
                    "needsRasterize": vector_element.get("needsRasterize"),
                    "hasSoftMask": vector_element.get("hasSoftMask"),
                    "fillType": vector_element.get("fillType"),
                    "patternType": vector_element.get("patternType"),
                    "patternImages": vector_element.get("patternImages"),
                    "_splitFrom": "stream",  # Mark as stream-split
                    "_splitIndex": sub_idx
                }
                
                # Check overlap for split vectors (if merging enabled)
                if self.enable_overlap_grouping:
                    merged = self._check_and_merge_overlap(sub_vector)
                    if merged:
                        continue  # Skip append if merged
                
                self.vectors.append(sub_vector)
        
        except Exception as e:
            logger.warning(f"[STREAM SPLIT] Failed to split vector: {e}")
            
            # Check overlap for unsplit vector (if merging enabled)
            if self.enable_overlap_grouping:
                merged = self._check_and_merge_overlap(vector_element)
                if not merged:
                    self.vectors.append(vector_element)
            else:
                self.vectors.append(vector_element)

    # --- main extraction method ---
    def extract(self, page) -> None:
        """
        Extract vectors from a PDF page.
        CRITICAL: This is the main extraction method from legacy code.
        """
        self._load_resources(page)

        try:
            ops = parse_content_stream(page)
        except Exception as e:
            logger.warning(f"Content stream parse error: {e}")
            return

        # DEBUG: Track unknown operators (especially marked content)
        unknown_ops = set()
        
        # Pre-computed string set for faster checking
        marked_content_ops_str = _MARKED_CONTENT_OPS_STR

        for inst in ops:
            op = str(inst.operator)
            
            # Fast-track numeric argument parsing to avoid overhead
            args: List[float] = []
            for x in inst.operands:
                try:
                    args.append(float(x))
                except Exception:
                    pass

            if op == _op_str(OP_SAVE_STATE):
                self.gstack.append(self.gstate.copy())
                # Save current clipping region on stack
                self.clipping_stack.append(self.active_clipping_region)
                # Save current clip stack depth (to restore on Q)
                self.clip_depth_stack.append(len(self.active_clip_stack))

            elif op == _op_str(OP_RESTORE_STATE):
                if self.gstack:
                    self.gstate = self.gstack.pop()
                # Restore clip stack to saved depth (pop clips added since matching q)
                if self.clip_depth_stack:
                    saved_depth = self.clip_depth_stack.pop()
                    while len(self.active_clip_stack) > saved_depth:
                        self.active_clip_stack.pop()
                # Restore clipping region from stack
                if self.clipping_stack:
                    self.active_clipping_region = self.clipping_stack.pop()

            elif op == _op_str(OP_CTM) and len(args) == 6:
                self.gstate.apply_ctm(*args)

            elif op == _op_str(OP_SET_GRAPHICS_STATE_PARAMS) and inst.operands:
                name = str(inst.operands[0]).lstrip("/")
                gs = self._extg.get(name)
                if gs is not None:
                    gs = self._deref(gs)
                    try:
                        if KEY_FILL_OPACITY in gs:
                            self.gstate.fill_opacity = float(gs[KEY_FILL_OPACITY])
                        if KEY_STROKE_OPACITY in gs:
                            self.gstate.stroke_opacity = float(gs[KEY_STROKE_OPACITY])
                        if KEY_BLEND_MODE in gs:
                            bm = gs[KEY_BLEND_MODE]
                            self.gstate.blend_mode = str(bm).lstrip("/") if bm else None
                        if KEY_SOFT_MASK in gs:
                            smask = gs[KEY_SOFT_MASK]
                            self.gstate.has_soft_mask = (smask and str(smask) != VAL_NONE)
                        if KEY_LINE_WIDTH in gs:
                            self.gstate.line_width = float(gs[KEY_LINE_WIDTH])
                        if KEY_LINE_CAP in gs:
                            self.gstate.line_cap = int(gs[KEY_LINE_CAP])
                        if KEY_LINE_JOIN in gs:
                            self.gstate.line_join = int(gs[KEY_LINE_JOIN])
                        if KEY_MITER_LIMIT in gs:
                            self.gstate.miter_limit = float(gs[KEY_MITER_LIMIT])
                    except Exception:
                        pass

            # colors
            elif op == _op_str(OP_SET_RGB_COLOR_STROKE) and len(args) == 3:
                self.gstate.stroke_color = (args[0], args[1], args[2])
                self.gstate.stroke_pattern = None
            elif op == _op_str(OP_SET_RGB_COLOR_FILL) and len(args) == 3:
                self.gstate.fill_color = (args[0], args[1], args[2])
                self.gstate.fill_pattern = None

            elif op == _op_str(OP_SET_GRAY_STROKE) and len(args) == 1:
                g = args[0]
                self.gstate.stroke_color = (g, g, g)
                self.gstate.stroke_pattern = None
            elif op == _op_str(OP_SET_GRAY_FILL) and len(args) == 1:
                g = args[0]
                self.gstate.fill_color = (g, g, g)
                self.gstate.fill_pattern = None

            elif op == _op_str(OP_SET_CMYK_COLOR_STROKE) and len(args) == 4:
                self.gstate.stroke_color = _cmyk_to_rgb(args[0], args[1], args[2], args[3])
                self.gstate.stroke_pattern = None
            elif op == _op_str(OP_SET_CMYK_COLOR_FILL) and len(args) == 4:
                self.gstate.fill_color = _cmyk_to_rgb(args[0], args[1], args[2], args[3])
                self.gstate.fill_pattern = None

            elif op in (_op_str(OP_SET_COLOR_STROKE), _op_str(OP_SET_COLOR_STROKE_N)):
                self._handle_color_set(is_stroke=True, inst=inst, args=args)
            elif op in (_op_str(OP_SET_COLOR_FILL), _op_str(OP_SET_COLOR_FILL_N)):
                self._handle_color_set(is_stroke=False, inst=inst, args=args)

            elif op == _op_str(OP_SET_LINE_WIDTH) and len(args) == 1:
                self.gstate.line_width = args[0]

            elif op == _op_str(OP_SET_LINE_CAP) and len(args) == 1:
                self.gstate.line_cap = int(args[0])

            elif op == _op_str(OP_SET_LINE_JOIN) and len(args) == 1:
                self.gstate.line_join = int(args[0])

            elif op == _op_str(OP_SET_MITER_LIMIT) and len(args) == 1:
                self.gstate.miter_limit = args[0]

            elif op == _op_str(OP_SET_DASH):
                try:
                    self.gstate.dash_array = [float(x) for x in inst.operands[0]] if inst.operands[0] else []
                    self.gstate.dash_phase = float(inst.operands[1])
                except Exception:
                    self.gstate.dash_array = []
                    self.gstate.dash_phase = 0.0

            elif op == _op_str(OP_SHADING) and inst.operands:
                name = str(inst.operands[0]).lstrip("/")
                sh = self._shadings.get(name)
                if sh is not None:
                    # sh paints the shading - emit as a rectangle using the shading pattern
                    parsed_shading = self._parse_shading(sh, pattern_matrix=None)
                    if parsed_shading:
                        # Use full page size
                        x, y, w, h = 0, 0, 1000, 1000  # Fallback
                        
                        # Emit shading as filled rect
                        self._emit_shading_rect(x, y, w, h, parsed_shading)

            elif op == _op_str(OP_DO_XOBJECT) and inst.operands:
                name = str(inst.operands[0]).lstrip("/")
                # Skip image XObjects (handled by image extractor)
                if self._is_image_xobject(name):
                    continue
                
                # Process Form XObjects (may contain opacity-affected vector graphics)
                xobj = self._xobjects.get(name)
                if xobj is not None:
                    try:
                        xobj = self._deref(xobj)
                        subtype = xobj.get(KEY_SUBTYPE)
                        if subtype and str(subtype).lstrip("/") == VAL_FORM.lstrip("/"):
                            # Extract vectors from Form XObject recursively
                            self._process_form_xobject(xobj, xobj_name=name)
                    except Exception as e:
                        logger.debug(f"Form XObject processing error: {e}")

            elif op in _PATH_OPS_STR:
                self._parse_path_segment(op, args)
            
            # Clipping path operators (W/W*) - establish clipping regions for logical grouping
            elif op in (_op_str(OP_CLIP), _op_str(OP_CLIP_EVEN_ODD)):
                # Clipping path established - tag all subsequent vectors with this region
                if self.current_subpaths or self.current_subpath:
                    # Finalize current path as subpaths
                    if self.current_subpath:
                        self.current_subpaths.append(self.current_subpath)
                        self.current_subpath = []
                    
                    # Compute bounding box of clipping path
                    paths_svg, bbox = self._build_paths_and_bbox(self.current_subpaths, False)
                    
                    # Create hash of clipping path to reuse IDs for identical paths
                    # This allows multiple q/W/Q blocks with same path to share a clipping region
                    clip_rule = "evenodd" if op == _op_str(OP_CLIP_EVEN_ODD) else "nonzero"
                    clip_hash = hashlib.sha256(f"{paths_svg}|{clip_rule}".encode()).hexdigest()[:16]
                    
                    # Check if we've seen this clipping path before
                    existing_region = None
                    for region in self.clipping_regions:
                        if region.get("hash") == clip_hash:
                            existing_region = region
                            break
                    
                    if existing_region:
                        # Reuse existing clipping region ID
                        clipping_region = existing_region
                    else:
                        # Create new clipping region with the actual path data
                        self.clipping_region_id += 1
                        clipping_region = {
                            "id": self.clipping_region_id,
                            "rule": clip_rule,
                            "bbox": bbox,
                            "hash": clip_hash,
                            "path": ' '.join(paths_svg) if paths_svg else None,  # Join all path segments
                        }
                        self.clipping_regions.append(clipping_region)
                    
                    self.active_clipping_region = clipping_region
                    
                    # Add to active clip stack (for nested clipping)
                    if clipping_region not in self.active_clip_stack:
                        self.active_clip_stack.append(clipping_region)
                    
                    # Reset path (clipping path is not painted)
                    self._reset_path()

            elif op in _PAINT_OPS_STR:
                if op == _op_str(OP_END_PATH):
                    # End path without painting
                    pass
                else:
                    has_stroke = op in _STROKE_OPS_STR
                    has_fill = op in _FILL_OPS_STR
                    fill_rule = "evenodd" if op in _EVEN_ODD_OPS_STR else "nonzero"
                    self._emit_vector(has_stroke, has_fill, fill_rule)
                self._reset_path()
            
            # DEBUG: Track unhandled operators
            # Note: Checking against strings here to avoid import bloat for every possible PDF op
            elif op not in {'RG', 'rg', 'G', 'g', 'K', 'k', 'SC', 'SCN', 'sc', 'scn', 'w', 'J', 'j', 'M', 'd', 'sh', 'Do', 'gs', 'Tf', 'Tm', 'Td', 'TD', 'T*', 'Tj', 'TJ', "'", '"', 'BT', 'ET', 'Tc', 'Tw', 'Tz', 'TL', 'Ts', 'Tr'}:
                if op in marked_content_ops_str:
                    # Already handled silently (or logged in debug if needed)
                    pass
                elif op not in unknown_ops:
                    unknown_ops.add(op)
                    logger.debug(f"[UNHANDLED_OP] {op} | Sample operands: {inst.operands}")
        
        # DEBUG: Report unknown operators at end
        if unknown_ops:
            logger.debug(f"[UNHANDLED_OPS] Found {len(unknown_ops)} unhandled operator types: {sorted(unknown_ops)}")
        
        # Post-processing: merge all overlapping vectors (catches non-consecutive overlaps)
        if self.enable_post_processing_merge and self.vectors:
            logger.debug(f"[POST-PROCESS] Starting full overlap merge (initial: {len(self.vectors)} vectors)")
            from utils.vector_grouping import post_process_merge_overlaps
            
            self.vectors = post_process_merge_overlaps(
                self.vectors,
                overlap_method=self.overlap_method,
                overlap_threshold=self.overlap_threshold,
                proximity_threshold=self.proximity_threshold
            )
            logger.debug(f"[POST-PROCESS] Completed (final: {len(self.vectors)} vectors)")