#!/usr/bin/env python3
"""
Content-aware bounding box calculation for vectors.

Calculates tight bounding boxes around actual path content and splits
large sparse vectors into smaller segments with minimal empty space.
"""

import re
from typing import List, Tuple, Optional

def calculate_path_segments_bounds(path_d: str) -> List[Tuple[float, float, float, float]]:
    """
    Calculate individual bounding boxes for each disconnected path segment.
    
    A path may contain multiple subpaths (separated by M commands after the first).
    Each subpath represents a potentially separate visual element.
    
    Returns: List of (min_x, min_y, max_x, max_y) for each segment
    """
    CMD_RE = re.compile(r'([MLHVCSQTAZ])\s*([^MLHVCSQTAZ]*)', re.IGNORECASE)
    NUM_RE = re.compile(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?')
    
    segments_bounds = []
    current_segment_points = []
    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    
    for match in CMD_RE.finditer(path_d):
        cmd = match.group(1).upper()
        coords_str = match.group(2).strip()
        
        if cmd == 'M':
            # New subpath - save previous segment if exists
            if current_segment_points:
                xs = [p[0] for p in current_segment_points]
                ys = [p[1] for p in current_segment_points]
                segments_bounds.append((min(xs), min(ys), max(xs), max(ys)))
                current_segment_points = []
            
            # Move to new position
            if coords_str:
                nums = NUM_RE.findall(coords_str)
                if len(nums) >= 2:
                    current_x = float(nums[0])
                    current_y = float(nums[1])
                    start_x, start_y = current_x, current_y
                    current_segment_points.append((current_x, current_y))
        
        elif cmd == 'L':
            if coords_str:
                nums = NUM_RE.findall(coords_str)
                for i in range(0, len(nums), 2):
                    if i + 1 < len(nums):
                        current_x = float(nums[i])
                        current_y = float(nums[i + 1])
                        current_segment_points.append((current_x, current_y))
        
        elif cmd == 'H':
            if coords_str:
                nums = NUM_RE.findall(coords_str)
                for num in nums:
                    current_x = float(num)
                    current_segment_points.append((current_x, current_y))
        
        elif cmd == 'V':
            if coords_str:
                nums = NUM_RE.findall(coords_str)
                for num in nums:
                    current_y = float(num)
                    current_segment_points.append((current_x, current_y))
        
        elif cmd == 'C':
            if coords_str:
                nums = NUM_RE.findall(coords_str)
                for i in range(0, len(nums), 6):
                    if i + 5 < len(nums):
                        # Add all control points and end point
                        current_segment_points.extend([
                            (float(nums[i]), float(nums[i+1])),
                            (float(nums[i+2]), float(nums[i+3])),
                            (float(nums[i+4]), float(nums[i+5]))
                        ])
                        current_x = float(nums[i+4])
                        current_y = float(nums[i+5])
        
        elif cmd == 'Z':
            # Close path - add start point
            current_segment_points.append((start_x, start_y))
            current_x, current_y = start_x, start_y
    
    # Save last segment
    if current_segment_points:
        xs = [p[0] for p in current_segment_points]
        ys = [p[1] for p in current_segment_points]
        segments_bounds.append((min(xs), min(ys), max(xs), max(ys)))
    
    return segments_bounds


def merge_nearby_bounds(
    bounds_list: List[Tuple[float, float, float, float]],
    merge_distance: float = 10.0
) -> List[Tuple[float, float, float, float]]:
    """
    Merge bounding boxes that are close to each other.
    
    Args:
        bounds_list: List of (min_x, min_y, max_x, max_y)
        merge_distance: Distance threshold for merging (in points)
    
    Returns:
        Merged list of bounding boxes
    """
    if not bounds_list:
        return []
    
    # Start with all bounds
    merged = list(bounds_list)
    changed = True
    
    while changed:
        changed = False
        new_merged = []
        used = set()
        
        for i, b1 in enumerate(merged):
            if i in used:
                continue
            
            # Try to merge with subsequent bounds
            current_bounds = b1
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                
                b2 = merged[j]
                
                # Check if bounds are close enough to merge
                if bounds_overlap_or_close(current_bounds, b2, merge_distance):
                    # Merge the bounds
                    current_bounds = (
                        min(current_bounds[0], b2[0]),
                        min(current_bounds[1], b2[1]),
                        max(current_bounds[2], b2[2]),
                        max(current_bounds[3], b2[3])
                    )
                    used.add(j)
                    changed = True
            
            new_merged.append(current_bounds)
            used.add(i)
        
        merged = new_merged
    
    return merged


def bounds_overlap_or_close(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
    distance: float
) -> bool:
    """Check if two bounding boxes overlap or are within distance of each other."""
    min_x1, min_y1, max_x1, max_y1 = b1
    min_x2, min_y2, max_x2, max_y2 = b2
    
    # Expand bounds by distance
    min_x1 -= distance
    min_y1 -= distance
    max_x1 += distance
    max_y1 += distance
    
    # Check for overlap
    return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)


def extract_subpath_from_svg(svg_content: str, target_bounds: Tuple[float, float, float, float]) -> Optional[str]:
    """
    Extract a portion of an SVG that falls within the target bounds.
    
    Args:
        svg_content: Full SVG markup
        target_bounds: (min_x, min_y, max_x, max_y) in absolute coordinates
    
    Returns:
        New SVG with adjusted viewBox and paths clipped to bounds
    """
    # Extract path d attribute
    path_match = re.search(r'<path\s+d="([^"]+)"', svg_content)
    if not path_match:
        return None
    
    path_d = path_match.group(1)
    
    # Extract all attributes
    attrs_match = re.search(r'<path\s+([^>]+)>', svg_content)
    if not attrs_match:
        return None
    
    attrs = attrs_match.group(1)
    
    # Calculate new viewBox
    min_x, min_y, max_x, max_y = target_bounds
    width = max_x - min_x
    height = max_y - min_y
    
    # Shift path coordinates to be relative to new viewBox
    CMD_RE = re.compile(r'([MLHVCSQTAZ])\s*([^MLHVCSQTAZ]*)', re.IGNORECASE)
    NUM_RE = re.compile(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?')
    
    new_path_parts = []
    for match in CMD_RE.finditer(path_d):
        cmd = match.group(1)
        coords_str = match.group(2).strip()
        
        if cmd == 'Z':
            new_path_parts.append('Z')
        elif cmd == 'H':
            if coords_str:
                x = float(coords_str) - min_x
                new_path_parts.append(f'H {x:.3f}')
        elif cmd == 'V':
            if coords_str:
                y = float(coords_str) - min_y
                new_path_parts.append(f'V {y:.3f}')
        elif coords_str:
            nums = NUM_RE.findall(coords_str)
            shifted = [
                float(nums[i]) - (min_x if i % 2 == 0 else min_y)
                for i in range(len(nums))
            ]
            new_path_parts.append(f'{cmd} ' + ' '.join(f'{n:.3f}' for n in shifted))
    
    new_path_d = ' '.join(new_path_parts)
    
    # Build new SVG
    new_svg = (
        f'<svg viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<path d="{new_path_d}" {attrs} />'
        f'</svg>'
    )
    
    return new_svg


# Test with sample path
if __name__ == "__main__":
    # Example: Large page-spanning path with corner markers
    test_path = "M 27.850 0.100 L 27.850 24.100 M 0.100 27.850 L 24.100 27.850 M 869.740 0.100 L 869.740 24.100 M 897.490 27.850 L 873.490 27.850"
    
    print("Testing path segment bounds calculation")
    print(f"Input path: {test_path}\n")
    
    segments = calculate_path_segments_bounds(test_path)
    print(f"Found {len(segments)} segments:")
    for i, (min_x, min_y, max_x, max_y) in enumerate(segments):
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        print(f"  Segment {i}: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
        print(f"             Size: {width:.1f} x {height:.1f} (area: {area:.0f})")
    
    print(f"\nMerging nearby segments (distance < 50pt)...")
    merged = merge_nearby_bounds(segments, merge_distance=50.0)
    print(f"Merged to {len(merged)} groups:")
    for i, (min_x, min_y, max_x, max_y) in enumerate(merged):
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        print(f"  Group {i}: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
        print(f"            Size: {width:.1f} x {height:.1f} (area: {area:.0f})")
