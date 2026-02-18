"""PDF transformation utilities for graphics operations."""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

MATRIX_EPSILON = 1e-9
ANGLE_TOLERANCE = 1e-6
NINETY_DEGREE_HEIGHT_FACTOR = 0.70

@dataclass
class Transformation:
    """Holds decomposed transformation matrix values."""
    rotation: float
    scaleX: float
    scaleY: float
    skewX: float
    skewY: float
    translateX: float
    translateY: float

# --- Core Transformation Functions ---
def decompose_ctm(ctm: List[float]) -> Transformation:
    """Decompose CTM matrix into transformation components.
    
    Args:
        ctm: 6-element CTM matrix [a, b, c, d, e, f]
    
    Returns:
        Transformation with rotation, scale, skew, and translation.
    """
    a, b, c, d, e, f = ctm
    
    # Translation is straightforward
    translateX = e
    translateY = f
    
    # Calculate rotation angle using numpy's arctan2 for proper quadrant handling
    rotation_radians = np.arctan2(b, a)
    rotation_degrees = float(np.degrees(rotation_radians))
    
    # Calculate scaling factors
    scaleX = float(np.sqrt(a * a + b * b))
    scaleY = float(np.sqrt(c * c + d * d))
    
    # Calculate skew angles
    if abs(scaleX) > MATRIX_EPSILON:
        skewX_component = (a * c + b * d) / (a * a + b * b)
        skewX_radians = np.arctan(skewX_component)
        skewX_degrees = float(np.degrees(skewX_radians))
    else:
        skewX_degrees = 0.0
    
    if abs(scaleY) > MATRIX_EPSILON:
        skewY_radians = np.arctan((a * c + b * d) / (c * c + d * d))
        skewY_degrees = float(np.degrees(skewY_radians))
    else:
        skewY_degrees = 0.0
    
    # Handle reflection (negative determinant) by flipping one of the scales
    determinant = a * d - b * c
    if determinant < 0:
        scaleX = -scaleX
    
    return Transformation(
        rotation=rotation_degrees,
        scaleX=scaleX,
        scaleY=scaleY,
        skewX=skewX_degrees,
        skewY=skewY_degrees,
        translateX=translateX,
        translateY=translateY
    )

def get_text_pivot_point(
    origin_x: float, origin_y: float, height: float, angle: float
) -> Tuple[float, float]:
    """Calculate anchor point for horizontal text block.
    
    Uses inverse logic for text normalizer to position rotated text baseline correctly.
    """
    if angle == 0:
        return origin_x, origin_y

    angle_rad = math.radians(angle)
    sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)
    dx = height * sin_a

    if abs(abs(angle) - 90) < ANGLE_TOLERANCE:
        dy = height * NINETY_DEGREE_HEIGHT_FACTOR
    else:
        dy = height * (1 - cos_a)

    return origin_x - dx, origin_y - dy

def get_image_pivot_point(
    origin_x: float, origin_y: float, height: float, angle: float
) -> Tuple[float, float]:
    """Calculate top-left anchor point for rotated image.
    
    Finds un-rotated pivot of rectangle rotated around bottom-left corner.
    """
    if angle == 0:
        return origin_x, origin_y + height

    angle_rad = math.radians(angle)
    sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)
    dx = height * sin_a
    dy = height * (1 - cos_a)

    return origin_x - dx, (origin_y + height) - dy

def apply_matrix_transform(x: float, y: float, ctm: List[float]) -> Tuple[float, float]:
    """Apply CTM transformation to a point.
    
    Args:
        x, y: Point coordinates
        ctm: 6-element CTM matrix [a, b, c, d, e, f]
    
    Returns:
        Transformed (x, y) coordinates
    """
    a, b, c, d, e, f = ctm
    tx = a * x + c * y + e
    ty = b * x + d * y + f
    return tx, ty

def calculate_image_bbox(ctm: List[float], page_height: float, mediabox_bottom: float = 0.0) -> dict:
    """Calculate image bounding box in Y-down coordinates.
    
    Transforms image unit square (0,0)-(1,1) using CTM.
    
    Args:
        ctm: Current Transformation Matrix [a, b, c, d, e, f]
        page_height: Page height for Y coordinate conversion
        mediabox_bottom: MediaBox bottom Y-coordinate (default: 0.0)
            pdfminer coordinates are in raw PDF space, must subtract this offset
            to align with pikepdf coordinates that are MediaBox-relative
    
    Returns:
        dict with x, y, width, height in Y-down coordinates
    """
    # Image unit square coordinates (PDF images are defined on unit square)
    unit_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    # Transform all corners using CTM
    transformed_coords = []
    for x, y in unit_coords:
        tx, ty = apply_matrix_transform(x, y, ctm)
        # Convert to frontend Y-down coordinate system
        ty_frontend = page_height - ty
        transformed_coords.append((tx, ty_frontend))
    
    # Find bounding rectangle
    x_coords = [coord[0] for coord in transformed_coords]
    y_coords = [coord[1] for coord in transformed_coords]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return {
        'x': min_x,
        'y': min_y,
        'width': max_x - min_x,
        'height': max_y - min_y
    }

def calculate_image_bbox_pdf_coords(ctm: List[float]) -> Tuple[float, float, float, float]:
    """Calculate image bounding box in PDF Y-up coordinates.
    
    Args:
        ctm: Current Transformation Matrix [a, b, c, d, e, f]
    
    Returns:
        Tuple of (x, y, width, height) in PDF coordinates
    """
    # Image unit square coordinates
    unit_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    # Transform all corners using CTM
    transformed_coords = []
    for x, y in unit_coords:
        tx, ty = apply_matrix_transform(x, y, ctm)
        transformed_coords.append((tx, ty))
    
    # Find bounding rectangle
    x_coords = [coord[0] for coord in transformed_coords]
    y_coords = [coord[1] for coord in transformed_coords]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return min_x, min_y, max_x - min_x, max_y - min_y