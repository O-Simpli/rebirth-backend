"""
Vector Grouping Utilities

Provides overlap detection and grouping logic for vector elements.
Used during stream-based extraction to identify related vectors.
"""

from typing import List, Tuple, Optional, TypedDict
import logging

logger = logging.getLogger(__name__)


class VectorDict(TypedDict, total=False):
    """Internal vector dictionary structure used during extraction.
    
    This matches the dict structure created by VectorExtractionDevice
    before conversion to PdfVectorElement Pydantic models.
    """
    id: str
    x: float
    y: float
    width: float
    height: float
    svgContent: str
    dataUri: str
    stroke: Optional[str]
    fill: Optional[str]
    strokeWidth: Optional[float]
    pathData: Optional[str]
    pathCount: int
    needsRasterize: bool
    hasSoftMask: bool
    opacity: float
    groupId: Optional[str]
    fillType: Optional[str]
    patternType: Optional[int]
    patternImages: Optional[List[dict]]


def calculate_overlap_area(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate the area of overlap between two bounding boxes.
    
    Args:
        bbox1: (x, y, width, height) of first box
        bbox2: (x, y, width, height) of second box
        
    Returns:
        Overlap area in square points (0 if no overlap)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection bounds
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    
    # Check if there's actual overlap
    if right <= left or bottom <= top:
        return 0.0
    
    return (right - left) * (bottom - top)


def calculate_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: (x, y, width, height) of first box
        bbox2: (x, y, width, height) of second box
        
    Returns:
        IoU ratio (0.0 to 1.0)
    """
    overlap = calculate_overlap_area(bbox1, bbox2)
    
    if overlap == 0:
        return 0.0
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - overlap
    
    return overlap / union if union > 0 else 0.0


def has_significant_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    method: str = "area",
    threshold: float = 0.1
) -> bool:
    """
    Check if two bounding boxes have significant overlap.
    
    Args:
        bbox1: (x, y, width, height) of first box
        bbox2: (x, y, width, height) of second box
        method: Overlap detection method
            - "area": Overlap as percentage of smaller vector (default)
            - "iou": Intersection over Union
            - "intersection": Any overlap above threshold area
        threshold: Minimum overlap required (interpretation depends on method)
            - "area": 0.1 = 10% of smaller vector must overlap
            - "iou": 0.1 = 10% IoU ratio
            - "intersection": 10 square points minimum overlap
            
    Returns:
        True if overlap is significant
    """
    overlap_area = calculate_overlap_area(bbox1, bbox2)
    
    if overlap_area == 0:
        return False
    
    if method == "intersection":
        # Simple threshold: any overlap above N square points
        return overlap_area >= threshold
    
    elif method == "iou":
        # Intersection over Union
        iou = calculate_iou(bbox1, bbox2)
        return iou >= threshold
    
    elif method == "area":
        # Percentage of smaller vector that overlaps
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        area1 = w1 * h1
        area2 = w2 * h2
        smaller_area = min(area1, area2)
        
        if smaller_area == 0:
            return False
        
        overlap_percentage = overlap_area / smaller_area
        return overlap_percentage >= threshold
    
    else:
        raise ValueError(f"Unknown overlap method: {method}")


def calculate_proximity(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate minimum distance between two bounding boxes.
    
    Returns 0 if boxes overlap, otherwise the minimum edge distance.
    
    Args:
        bbox1: (x, y, width, height) of first box
        bbox2: (x, y, width, height) of second box
        
    Returns:
        Minimum distance in points (0 if overlapping)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate box edges
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Check for overlap
    if not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1):
        return 0.0  # Overlapping
    
    # Calculate horizontal distance
    if right1 < left2:
        dx = left2 - right1
    elif right2 < left1:
        dx = left1 - right2
    else:
        dx = 0.0
    
    # Calculate vertical distance
    if bottom1 < top2:
        dy = top2 - bottom1
    elif bottom2 < top1:
        dy = top1 - bottom2
    else:
        dy = 0.0
    
    # Return Euclidean distance
    return (dx * dx + dy * dy) ** 0.5


def is_contained_within(
    inner_bbox: Tuple[float, float, float, float],
    outer_bbox: Tuple[float, float, float, float],
    tolerance: float = 1.0
) -> bool:
    """
    Check if one bounding box is fully contained within another.
    
    Args:
        inner_bbox: (x, y, width, height) of potentially contained box
        outer_bbox: (x, y, width, height) of potentially containing box
        tolerance: Allow slight overflow (in points)
        
    Returns:
        True if inner box is contained within outer box
    """
    ix, iy, iw, ih = inner_bbox
    ox, oy, ow, oh = outer_bbox
    
    # Calculate bounds
    inner_left, inner_right = ix, ix + iw
    inner_top, inner_bottom = iy, iy + ih
    
    outer_left, outer_right = ox, ox + ow
    outer_top, outer_bottom = oy, oy + oh
    
    # Check containment with tolerance
    return (
        inner_left >= outer_left - tolerance and
        inner_right <= outer_right + tolerance and
        inner_top >= outer_top - tolerance and
        inner_bottom <= outer_bottom + tolerance
    )


def calculate_union_bounds(
    bboxes: List[Tuple[float, float, float, float]]
) -> Tuple[float, float, float, float]:
    """
    Calculate the union bounding box of multiple boxes.
    
    Args:
        bboxes: List of (x, y, width, height) tuples
        
    Returns:
        Union bounding box as (x, y, width, height)
    """
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    
    # Find extremes
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for x, y, w, h in bboxes:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def find_overlapping_vectors(
    new_vector: VectorDict,
    recent_vectors: List[VectorDict],
    overlap_method: str = "area",
    overlap_threshold: float = 0.1,
    proximity_threshold: Optional[float] = None
) -> List[int]:
    """
    Find indices of recent vectors that overlap with new vector.
    
    Args:
        new_vector: Vector dict with x, y, width, height
        recent_vectors: List of recent vector dicts to check against
        overlap_method: Method for overlap detection ("area", "iou", "intersection")
        overlap_threshold: Threshold for overlap significance
        proximity_threshold: If set, also group vectors within this distance (points)
        
    Returns:
        List of indices in recent_vectors that overlap with new_vector
    """
    overlapping_indices = []
    
    new_bbox = (
        new_vector.get('x', 0),
        new_vector.get('y', 0),
        new_vector.get('width', 0),
        new_vector.get('height', 0)
    )
    
    for i, vec in enumerate(recent_vectors):
        vec_bbox = (
            vec.get('x', 0),
            vec.get('y', 0),
            vec.get('width', 0),
            vec.get('height', 0)
        )
        
        # Check overlap
        if has_significant_overlap(new_bbox, vec_bbox, overlap_method, overlap_threshold):
            overlapping_indices.append(i)
            continue
        
        # Check proximity if enabled
        if proximity_threshold is not None:
            distance = calculate_proximity(new_bbox, vec_bbox)
            if distance <= proximity_threshold:
                overlapping_indices.append(i)
    
    return overlapping_indices


def merge_vectors(vectors: List[VectorDict]) -> VectorDict:
    """
    Merge multiple overlapping vectors into a single vector element.
    
    Combines SVG paths and computes union bounding box.
    Generates a new unique ID for the merged vector.
    
    Args:
        vectors: List of vectors to merge (must have at least one)
        
    Returns:
        Merged vector with combined SVG, union bounds, and new unique ID
    """
    import re
    from processors.vector_extraction_device import _stable_id
    
    if not vectors:
        raise ValueError("Cannot merge empty vector list")
    
    if len(vectors) == 1:
        return vectors[0].copy()
    
    # Calculate union bounds
    min_x = min(v['x'] for v in vectors)
    min_y = min(v['y'] for v in vectors)
    max_x = max(v['x'] + v['width'] for v in vectors)
    max_y = max(v['y'] + v['height'] for v in vectors)
    
    union_width = max_x - min_x
    union_height = max_y - min_y
    
    # Combine SVG paths
    merged_svg = f'<svg viewBox="0 0 {union_width:.2f} {union_height:.2f}" xmlns="http://www.w3.org/2000/svg">'
    merged_svg += '<g>'
    
    # Add each vector's SVG content with position adjustment
    for v in vectors:
        offset_x = v['x'] - min_x
        offset_y = v['y'] - min_y
        svg_content = v.get('svgContent', '')
        
        if not svg_content:
            continue
        
        # Extract content from SVG tag if present
        if '<svg' in svg_content:
            match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
            if match:
                svg_content = match.group(1)
        
        # Wrap in group with transform if needed
        if abs(offset_x) > 0.01 or abs(offset_y) > 0.01:
            merged_svg += f'<g transform="translate({offset_x:.2f},{offset_y:.2f})">{svg_content}</g>'
        else:
            merged_svg += svg_content
    
    merged_svg += '</g></svg>'
    
    # Generate new unique ID for merged vector using all source IDs
    # Use existing _stable_id helper from vector_extraction_device
    source_ids = sorted([v.get('id', '') for v in vectors])
    new_id = _stable_id("merged", *source_ids, min_x, min_y, union_width, union_height)
    
    # Use properties from first vector as base
    merged = vectors[0].copy()
    merged.update({
        'id': new_id,  # New unique ID
        'x': min_x,
        'y': min_y,
        'width': union_width,
        'height': union_height,
        'svgContent': merged_svg,
        'pathData': None  # Clear pathData for merged vectors
    })
    
    # Remove groupId if present
    merged.pop('groupId', None)
    
    return merged


def post_process_merge_overlaps(
    vectors: List[VectorDict],
    overlap_method: str = "area",
    overlap_threshold: float = 0.1,
    proximity_threshold: Optional[float] = None
) -> List[VectorDict]:
    """
    Post-process all vectors to merge non-consecutive overlaps missed during streaming.
    
    This performs a full O(N²) check to find and merge ALL overlapping vectors,
    not just those within the stream window. Use this to catch overlaps between
    vectors that were emitted far apart (e.g., layered icons, complex graphics).
    
    Args:
        vectors: List of all extracted vectors
        overlap_method: "area", "iou", or "intersection"
        overlap_threshold: Significance threshold
        proximity_threshold: Optional distance threshold
        
    Returns:
        New list with overlapping vectors merged
    """
    if not vectors:
        return []
    
    # Build adjacency graph of overlapping vectors
    n = len(vectors)
    overlaps = set()  # Set of (i, j) pairs where i < j
    
    for i in range(n):
        for j in range(i + 1, n):
            bbox1 = (vectors[i]['x'], vectors[i]['y'], vectors[i]['width'], vectors[i]['height'])
            bbox2 = (vectors[j]['x'], vectors[j]['y'], vectors[j]['width'], vectors[j]['height'])
            
            if has_significant_overlap(bbox1, bbox2, overlap_method, overlap_threshold):
                overlaps.add((i, j))
            elif proximity_threshold is not None:
                distance = calculate_proximity(bbox1, bbox2)
                if distance <= proximity_threshold:
                    overlaps.add((i, j))
    
    if not overlaps:
        return vectors  # No overlaps found
    
    # Find connected components using Union-Find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union overlapping vectors
    for i, j in overlaps:
        union(i, j)
    
    # Group vectors by component
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Merge each group
    merged_vectors = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            # Single vector, keep as-is
            merged_vectors.append(vectors[group_indices[0]])
        else:
            # Multiple vectors, merge them
            group_vectors = [vectors[i] for i in group_indices]
            merged = merge_vectors(group_vectors)
            merged_vectors.append(merged)
    
    return merged_vectors


def assign_group_id(
    vector: VectorDict,
    overlapping_vectors: List[VectorDict],
    next_group_id: int
) -> Tuple[str, int]:
    """
    Assign group ID to vector based on overlapping vectors.
    
    If vector overlaps with vectors that already have group IDs,
    use the existing group ID. Otherwise, assign a new one.
    
    Args:
        vector: Vector dict to assign group ID to
        overlapping_vectors: List of overlapping vector dicts
        next_group_id: Next available group ID number
        
    Returns:
        Tuple of (assigned_group_id, next_available_group_id)
    """
    # Check if any overlapping vectors already have a group ID
    existing_groups = set()
    for vec in overlapping_vectors:
        if 'groupId' in vec and vec['groupId']:
            existing_groups.add(vec['groupId'])
    
    if existing_groups:
        # Use the first existing group ID (prefer lower IDs)
        group_id = sorted(existing_groups)[0]
        return (group_id, next_group_id)
    else:
        # Assign new group ID
        group_id = f"group_{next_group_id}"
        next_group_id += 1
        return (group_id, next_group_id)
