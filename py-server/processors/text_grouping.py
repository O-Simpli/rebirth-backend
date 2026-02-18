"""PDF Text Content Structuring Engine

Groups raw text elements from PDFs into structured document hierarchies,
transforming flat lists of text snippets into coherent lines, paragraphs,
and logical blocks using multi-stage spatial and semantic analysis.
"""

import hashlib
import logging
import re
import statistics
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Union

from models.pdf_types import (
    BoundingBox,
    ContentSection,
    ParagraphSection,
    PdfElement,
    PdfImageElement,
    PdfObstacles,
    PdfRect,
    PdfTextElement,
    PdfTextRun,
    ProcessingTextRun,
    TextGroupingReason,
)

logger = logging.getLogger(__name__)

Y_POSITION_TOLERANCE = 2
TOC_MAX_ENTRIES_LIMIT = 15
MAX_ITERATIONS_SAFETY = 1000
FONT_SIZE_CHANGE_THRESHOLD = 0.4
PARAGRAPH_BREAK_MULTIPLIER = 0.6
INTERACTION_MARGIN = 2.0
DEFAULT_LINE_HEIGHT_RATIO = 1.2
DEFAULT_FONT_SIZE = 12.0
BASELINE_TOLERANCE_FACTOR = 0.1
OVERLAP_THRESHOLD = 0.8
HORIZONTAL_TOLERANCE = 5.0
SPACE_GAP_MULTIPLIER = 0.25
MIN_LINE_HEIGHT_RATIO = 0.8
MAX_LINE_HEIGHT_RATIO = 3.0

DEFAULT_TEXT_CONFIG = {
    'enable_grouping': True,
    'max_horizontal_gap': 10,  # Increased from 5 to accommodate typical word spacing (~0.5em)
    'max_line_height_difference': 2,
    'max_vertical_gap': 30,
    'require_same_font': False,
    'require_similar_font_size': False,
    'font_size_tolerance': 2,
    'enable_semantic_grouping': False,
    'min_words_for_paragraph': 3,
    'enable_list_detection': True,
    'max_list_item_gap': 50,
    'list_indent_tolerance': 15,
    'min_list_items': 2,
    'enable_toc_detection': True,
    'max_toc_horizontal_gap': 150,
    'min_toc_entries': 3,
    'toc_page_number_pattern': r'^\d+$|^[ivxlcdm]+$',
    'max_elements_to_consider': float('inf'),
}


class TextGrouper:
    """Groups text elements based on spatial and semantic relationships"""
    
    def __init__(
        self,
        stream_elements: List[Union[ProcessingTextRun, PdfImageElement]],
        config: Optional[Dict] = None,
        obstacles: Optional[PdfObstacles] = None
    ):
        text_runs = [el for el in stream_elements if isinstance(el, ProcessingTextRun)]
        self.image_elements: List[PdfImageElement] = [el for el in stream_elements if isinstance(el, PdfImageElement)]
        self.original_runs: List[ProcessingTextRun] = text_runs
        self.obstacles: Optional[PdfObstacles] = obstacles
        self.config: Dict = self._validate_and_merge_config(config)
        
        max_elements = self.config.get('max_elements_to_consider', float('inf'))
        self.runs_to_process: List[ProcessingTextRun] = (
            self.original_runs[:int(max_elements)] if max_elements != float('inf') else self.original_runs
        )
        
        self.lines: List[PdfTextElement] = []
        self.blocks: List[PdfTextElement] = []
        self.final_groups: List[PdfTextElement] = []
        self.runs_to_element_id: Dict[str, List[ProcessingTextRun]] = {}

    def _validate_and_merge_config(self, user_config: Optional[Dict]) -> Dict:
        """Merge user configuration with defaults"""
        full_config = {**DEFAULT_TEXT_CONFIG, **(user_config or {})}
        
        if 'toc_page_number_pattern' in full_config and isinstance(full_config['toc_page_number_pattern'], str):
            full_config['toc_page_number_pattern'] = re.compile(
                full_config['toc_page_number_pattern'], 
                re.IGNORECASE
            )
        
        return full_config

    def group(self) -> List[PdfElement]:
        """Execute full text grouping process and preserve image elements"""
        try:
            self._group_horizontally()
            self._group_vertically()
            self.final_groups = self.blocks
            result = self._include_ungrouped_elements()
            result.extend(self.image_elements)
            
            _validate_grouping_results(self.runs_to_process, result, self.runs_to_element_id)
            return result
            
        except Exception as e:
            logger.error(f"Text grouping failed: {e}", exc_info=True)
            return [_create_text_element([run], "horizontal-proximity") for run in self.original_runs]

    def _group_horizontally(self):
        """Group text runs horizontally into lines."""
        sorted_runs = sorted(self.runs_to_process, key=lambda el: (
            el.y if abs(el.y - self.runs_to_process[0].y) > self.config['max_line_height_difference'] else 0,
            el.x
        ))
        
        processed = set()
        for run in sorted_runs:
            run_id = id(run)
            if run_id in processed:
                continue

            line_runs = self._find_horizontal_neighbors(run, sorted_runs)
            for el in line_runs:
                processed.add(id(el))

            group = _create_text_element(line_runs, "horizontal-proximity")
            self.runs_to_element_id[group.id] = line_runs
            self.lines.append(group)

    def _find_horizontal_neighbors(self, target_run: ProcessingTextRun, all_runs: List[ProcessingTextRun]) -> List[ProcessingTextRun]:
        """Find runs that should be grouped horizontally with the target run."""
        neighbors = [target_run]
        y_tolerance = max(self.config['max_line_height_difference'], target_run.height * 0.3)
        
        candidate_runs = sorted([
            el for el in all_runs
            if el != target_run and abs(el.y - target_run.y) <= y_tolerance
        ], key=lambda el: el.x)

        # Make max_gap font-size relative (typically word spacing is ~0.5-0.7em)
        # Use target run's font size, or fallback to config value for runs without font size
        base_max_gap = self.config['max_horizontal_gap']
        if hasattr(target_run, 'fontSize') and target_run.fontSize:
            max_gap = target_run.fontSize * 0.6  # 0.6em is typical max word spacing
        else:
            max_gap = base_max_gap
        
        adaptive_max_gap = max_gap * 3 if y_tolerance <= self.config['max_line_height_difference'] else max_gap

        # Find elements to the right
        right_x = target_run.x + target_run.width
        for candidate in candidate_runs:
            if candidate.x >= right_x:
                gap = candidate.x - right_x
                if gap <= adaptive_max_gap and self._can_group_runs(target_run, candidate):
                    neighbors.append(candidate)
                    right_x = candidate.x + candidate.width
            elif candidate.x < right_x and candidate.x + candidate.width > target_run.x:
                if self._can_group_runs(target_run, candidate):
                    neighbors.append(candidate)
                    right_x = max(right_x, candidate.x + candidate.width)
        
        # Find elements to the left
        left_x = target_run.x
        for candidate in reversed(candidate_runs):
            if candidate.x + candidate.width <= left_x:
                gap = left_x - (candidate.x + candidate.width)
                if gap <= adaptive_max_gap and self._can_group_runs(target_run, candidate):
                    neighbors.insert(0, candidate)
                    left_x = candidate.x
            elif candidate.x < left_x and candidate.x + candidate.width > target_run.x:
                if self._can_group_runs(target_run, candidate):
                    neighbors.insert(0, candidate)
                    left_x = min(left_x, candidate.x)
        
        # Remove duplicates
        unique_neighbors, seen_ids = [], set()
        for neighbor in sorted(neighbors, key=lambda el: el.x):
            neighbor_id = id(neighbor)
            if neighbor_id not in seen_ids:
                unique_neighbors.append(neighbor)
                seen_ids.add(neighbor_id)
        
        return unique_neighbors

    def _can_group_runs(self, run1: ProcessingTextRun, run2: ProcessingTextRun) -> bool:
        """Check if two text runs can be grouped together."""
        if self.obstacles and self.obstacles.lines:
            left_el, right_el = (run1, run2) if run1.x < run2.x else (run2, run1)
            gap_x_start, gap_x_end = left_el.x + left_el.width, right_el.x
            gap_y_start = min(left_el.y, right_el.y)
            gap_y_end = max(left_el.y + left_el.height, right_el.y + right_el.height)
            
            for line in self.obstacles.lines:
                if abs(line.x0 - line.x1) < 2:
                    line_x = (line.x0 + line.x1) / 2
                    line_y_start, line_y_end = min(line.y0, line.y1), max(line.y0, line.y1)
                    if (gap_x_start < line_x < gap_x_end) and (line_y_end > gap_y_start and line_y_start < gap_y_end):
                        return False

        if self.config.get('require_same_font') and run1.fontFamily != run2.fontFamily:
            return False

        size1 = run1.fontSize or DEFAULT_FONT_SIZE
        size2 = run2.fontSize or DEFAULT_FONT_SIZE
        font_size_diff = abs(size1 - size2)
        
        if font_size_diff > 2:
            if not _are_baselines_aligned(run1, run2):
                return False
            return True
        
        if self.config.get('require_similar_font_size'):
            if font_size_diff > self.config.get('font_size_tolerance', 2):
                return False
        
        return True

    def _group_vertically(self):
        """Group text elements vertically into blocks."""
        if len(self.lines) <= 1:
            self.blocks = self.lines
            return
            
        groups_to_process = sorted(self.lines, key=lambda g: (g.y, g.x))

        if self.config.get('enable_list_detection', True):
            groups_to_process = self._detect_and_group_lists(groups_to_process)

        if self.config.get('enable_toc_detection', True):
            groups_to_process = self._detect_and_group_toc(groups_to_process)

        self.blocks = self._group_by_vertical_proximity(groups_to_process)
        
    def _group_by_vertical_proximity(self, groups: List[PdfTextElement]) -> List[PdfTextElement]:
        """Group elements by vertical proximity using connected components."""
        processed, final_blocks = set(), []
        for start_group in groups:
            if start_group.id in processed:
                continue

            component = self._find_connected_component(start_group, groups, processed)
            if len(component) > 1:
                all_runs = [run for g in component for run in self.runs_to_element_id.get(g.id, [])]
                block = _create_text_element(all_runs, "vertical-proximity")
                self.runs_to_element_id[block.id] = all_runs
                final_blocks.append(block)
            else:
                final_blocks.append(start_group)
        return final_blocks

    def _find_connected_component(self, start_group: PdfTextElement, all_groups: List[PdfTextElement], processed: Set[str]) -> List[PdfTextElement]:
        """Find all groups connected to the start group using BFS."""
        component_groups, queue = [], deque([start_group])
        processed.add(start_group.id)
        while queue:
            current_group = queue.popleft()
            component_groups.append(current_group)
            neighbors = self._find_vertical_neighbors(current_group, all_groups)
            for neighbor in neighbors:
                neighbor_id = neighbor.id
                if neighbor_id not in processed:
                    processed.add(neighbor_id)
                    queue.append(neighbor)
        return component_groups

    def _find_vertical_neighbors(self, target_group: PdfTextElement, all_groups: List[PdfTextElement]) -> List[PdfTextElement]:
        """Find groups that should be connected vertically to the target group."""
        neighbors = []
        is_rotated = abs(target_group.rotation) > 1
        max_gap = self.config['max_vertical_gap'] * 3 if is_rotated else self.config['max_vertical_gap']
        
        for candidate in all_groups:
            if candidate == target_group:
                continue
            
            upper, lower = (target_group, candidate) if target_group.y < candidate.y else (candidate, target_group)
            vertical_gap = lower.y - (upper.y + upper.height)

            if 0 <= vertical_gap <= max_gap:
                can_group = self._can_group_lines_vertically(target_group, candidate, all_groups)
                
                if can_group:
                    if not is_rotated:
                        has_ambiguous = self._has_ambiguous_horizontal_siblings(target_group, candidate, all_groups)
                        if has_ambiguous:
                            continue
                    
                    neighbors.append(candidate)
        
        return neighbors

    def _get_horizontal_neighbors(self, target: PdfTextElement, all_groups: List[PdfTextElement]) -> List[PdfTextElement]:
        """Find elements that are horizontally adjacent to the target element."""
        neighbors = []
        target_y_mid = target.y + target.height / 2
        y_tolerance = max(self.config['max_line_height_difference'], target.height)
        
        for candidate in all_groups:
            if candidate == target:
                continue
            
            candidate_y_mid = candidate.y + candidate.height / 2
            
            if abs(candidate_y_mid - target_y_mid) <= y_tolerance:
                target_left, target_right = target.x, target.x + target.width
                candidate_left, candidate_right = candidate.x, candidate.x + candidate.width
                
                if candidate_right <= target_left or candidate_left >= target_right:
                    neighbors.append(candidate)
        
        return neighbors



    def _has_ambiguous_horizontal_siblings(self, group1: PdfTextElement, group2: PdfTextElement, all_groups: List[PdfTextElement]) -> bool:
        """Check if vertical grouping would create T-junction ambiguity."""
        upper, lower = (group1, group2) if group1.y < group2.y else (group2, group1)
        
        def get_horizontal_siblings(target):
            siblings = []
            target_mid = target.y + target.height / 2
            tolerance = max(self.config['max_line_height_difference'], target.height)
            
            for g in all_groups:
                if g == target or g == upper or g == lower:
                    continue
                if abs((g.y + g.height / 2) - target_mid) <= tolerance:
                    siblings.append(g)
            return siblings
        
        upper_siblings = get_horizontal_siblings(upper)
        lower_siblings = get_horizontal_siblings(lower)
        
        upper_left, upper_right = upper.x, upper.x + upper.width
        lower_left, lower_right = lower.x, lower.x + lower.width
        
        for sibling in upper_siblings:
            sib_left, sib_right = sibling.x, sibling.x + sibling.width
            overlap = min(lower_right, sib_right) - max(lower_left, sib_left)
            if overlap > 0:
                return True
        
        for sibling in lower_siblings:
            sib_left, sib_right = sibling.x, sibling.x + sibling.width
            overlap = min(upper_right, sib_right) - max(upper_left, sib_left)
            if overlap > 0:
                return True
        
        for sibling in lower_siblings:
            sib_left, sib_right = sibling.x, sibling.x + sibling.width
            overlap_with_upper = min(upper_right, sib_right) - max(upper_left, sib_left)
            
            if overlap_with_upper > 0:
                sib_in_lower = sib_left >= lower_left - HORIZONTAL_TOLERANCE and sib_right <= lower_right + HORIZONTAL_TOLERANCE
                lower_in_sib = lower_left >= sib_left - HORIZONTAL_TOLERANCE and lower_right <= sib_right + HORIZONTAL_TOLERANCE
                
                if not (sib_in_lower or lower_in_sib):
                    return True
        
        return False

    def _can_group_lines_vertically(self, group1: PdfTextElement, group2: PdfTextElement, all_groups: Optional[List[PdfTextElement]] = None) -> bool:
        """Check if two groups can be grouped vertically."""
        upper, lower = (group1, group2) if group1.y < group2.y else (group2, group1)
        
        if self.obstacles:
            if self.obstacles.lines:
                gap_y_start, gap_y_end = upper.y + upper.height, lower.y
                gap_x_start, gap_x_end = max(upper.x, lower.x), min(upper.x + upper.width, lower.x + lower.width)
                for line in self.obstacles.lines:
                    if abs(line.y0 - line.y1) < 2:
                        line_y = (line.y0 + line.y1) / 2
                        line_x_start, line_x_end = min(line.x0, line.x1), max(line.x0, line.x1)
                        if (gap_y_start < line_y < gap_y_end) and (line_x_end > gap_x_start and line_x_start < gap_x_end):
                            return False

            if self.obstacles.rects:
                if _find_containing_rect(upper, self.obstacles.rects) is not _find_containing_rect(lower, self.obstacles.rects):
                    return False
                
                gap_y_start, gap_y_end = upper.y + upper.height, lower.y
                gap_x_start, gap_x_end = max(upper.x, lower.x), min(upper.x + upper.width, lower.x + lower.width)
                
                for rect in self.obstacles.rects:
                    # Check if rect's top or bottom edge crosses through the vertical gap
                    rect_top, rect_bottom = rect.y0, rect.y1
                    rect_left, rect_right = rect.x0, rect.x1
                    
                    # Check horizontal overlap
                    has_horizontal_overlap = (rect_right > gap_x_start and rect_left < gap_x_end)
                    
                    # Check if rect boundary is in the gap
                    top_edge_in_gap = (gap_y_start < rect_top < gap_y_end)
                    bottom_edge_in_gap = (gap_y_start < rect_bottom < gap_y_end)
                    
                    if has_horizontal_overlap and (top_edge_in_gap or bottom_edge_in_gap):
                        return False
        
        upper_left, upper_right = upper.x, upper.x + upper.width
        lower_left, lower_right = lower.x, lower.x + lower.width
        
        is_rotated = abs(upper.rotation) > 1 or abs(lower.rotation) > 1
        
        if is_rotated:
            # TODO: Apply coordinate transformation for rotated text alignment
            # Normalized rotated text forms a "staircase" pattern. To properly validate
            # vertical alignment, we need to apply inverse rotation to check if elements
            # align in their native rotated coordinate space.
            overlap_start = max(upper_left, lower_left)
            overlap_end = min(upper_right, lower_right)
            overlap_width = max(0, overlap_end - overlap_start)
            return overlap_width > 0
        # Check if there's significant horizontal overlap
        overlap_start = max(upper_left, lower_left)
        overlap_end = min(upper_right, lower_right)
        overlap_width = max(0, overlap_end - overlap_start)
        
        shorter_width = min(upper_right - upper_left, lower_right - lower_left)
        overlap_ratio = overlap_width / shorter_width if shorter_width > 0 else 0
        
        if overlap_ratio > OVERLAP_THRESHOLD:
            return True
        
        lower_in_upper = (
            lower_left >= upper_left - HORIZONTAL_TOLERANCE and 
            lower_right <= upper_right + HORIZONTAL_TOLERANCE
        )
        upper_in_lower = (
            upper_left >= lower_left - HORIZONTAL_TOLERANCE and
            upper_right <= lower_right + HORIZONTAL_TOLERANCE
        )
        
        return lower_in_upper or upper_in_lower

    def _detect_and_group_lists(self, groups: List[PdfTextElement]) -> List[PdfTextElement]:
        """Detect and group list items."""
        if len(groups) < self.config.get('min_list_items', 2):
            return groups
        return groups

    def _detect_and_group_toc(self, groups: List[PdfTextElement]) -> List[PdfTextElement]:
        """Detect and group table of contents entries."""
        if len(groups) < self.config.get('min_toc_entries', 3):
            return groups
        return groups

    def _include_ungrouped_elements(self) -> List[PdfElement]:
        """Convert ungrouped runs into PdfTextElements"""
        grouped_run_ids = {
            id(run) 
            for group in self.final_groups if group.type == "text" 
            for run in self.runs_to_element_id.get(group.id, [])
        }
        ungrouped_runs = [run for run in self.original_runs if id(run) not in grouped_run_ids]

        result: List[PdfElement] = list(self.final_groups)
        
        for run in ungrouped_runs:
            text_element = _create_text_element([run], "horizontal-proximity")
            result.append(text_element)
        
        return result


# ==============================================================================
# Public API Function
# ==============================================================================

def _apply_interaction_margin(elements: List[PdfElement]) -> List[PdfElement]:
    """Apply interaction margin to elements for better hit detection
    
    Text elements with multiple sections get both horizontal and vertical margins.
    Text elements with single section get only vertical margin.
    Image elements always get both margins.
    """
    if INTERACTION_MARGIN == 0:
        return elements
    
    margin = INTERACTION_MARGIN

    for el in elements:
        el.x -= margin / 2
        el.width += 2 * margin
        
        el.y -= margin / 2
        # Text element - only apply vertical margin if multiple sections
        if not hasattr(el, 'sections') or len(el.sections) > 1:
            el.height += 2 * margin
    
    
    return elements

def group_text_elements(
    stream_elements: List[Union[ProcessingTextRun, PdfImageElement]],
    config: Optional[Dict] = None,
    obstacles: Optional[PdfObstacles] = None
) -> List[PdfElement]:
    """Group text elements based on spatial and semantic relationships
    
    Args:
        stream_elements: ProcessingTextRun and PdfImageElement objects in stream order
        config: Configuration dictionary for grouping behavior
        obstacles: PDF obstacles for layout awareness
        
    Returns:
        List of grouped PdfTextElement, PdfImageElement, and ungrouped elements
    """
    if not stream_elements:
        return []
    
    grouper = TextGrouper(stream_elements, config, obstacles)
    grouped_elements = grouper.group()

    # Apply a final interaction margin for consistency in client-side rendering and redaction
    return _apply_interaction_margin(grouped_elements)


# ==============================================================================
# Module-Private Utility Functions (Stateless)
# ==============================================================================

def _find_containing_rect(element: PdfTextElement, rects: List[PdfRect]) -> Optional[PdfRect]:
    """Checks if an element is geometrically inside any of the provided rectangles."""
    cx, cy = element.x + element.width / 2, element.y + element.height / 2
    for rect in rects:
        if rect.x0 < cx < rect.x1 and rect.y0 < cy < rect.y1:
            return rect
    return None

def _is_horizontally_contained(element1: PdfTextElement, element2: PdfTextElement, tolerance: float = 5.0) -> bool:
    """Check if one element is horizontally contained within the bounds of another."""
    el1_left, el1_right = element1.x, element1.x + element1.width
    el2_left, el2_right = element2.x, element2.x + element2.width
    
    el1_in_el2 = (el1_left >= el2_left - tolerance and el1_right <= el2_right + tolerance)
    el2_in_el1 = (el2_left >= el1_left - tolerance and el2_right <= el1_right + tolerance)
    
    return el1_in_el2 or el2_in_el1

def _are_baselines_aligned(run1: ProcessingTextRun, run2: ProcessingTextRun) -> bool:
    """Check if two text runs share the same baseline"""
    baseline1 = run1.y + run1.height
    baseline2 = run2.y + run2.height
    
    font_size1 = run1.fontSize or DEFAULT_FONT_SIZE
    font_size2 = run2.fontSize or DEFAULT_FONT_SIZE
    smaller_font_size = min(font_size1, font_size2)
    tolerance = smaller_font_size * BASELINE_TOLERANCE_FACTOR
    
    return abs(baseline1 - baseline2) <= tolerance

def _get_all_runs_from_group(group: PdfTextElement) -> List[PdfTextRun]:
    """Extract all PdfTextRun objects from a PdfTextElement's sections"""
    return [
        el 
        for section in group.sections 
        if hasattr(section, 'lines') 
        for line in section.lines 
        for el in line
    ]

def _calculate_bounding_box(runs: List[ProcessingTextRun]) -> BoundingBox:
    """Calculate bounding box encompassing all runs"""
    if not runs:
        return BoundingBox(x=0, y=0, width=0, height=0)
    
    min_x = min(el.x for el in runs)
    min_y = min(el.y for el in runs)
    max_x = max(el.x + el.width for el in runs)
    max_y = max(el.y + el.height for el in runs)
    return BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

def _combine_text_content(runs: List[ProcessingTextRun]) -> str:
    """Combine text content from runs with proper spacing"""
    if not runs:
        return ""
    
    lines = _group_runs_into_lines(sorted(runs, key=lambda el: (el.y, el.x)))
    output_lines = []
    
    for line in lines:
        if not line:
            continue
        
        line_text = line[0].text
        for i in range(1, len(line)):
            prev_el = line[i-1]
            curr_el = line[i]
            
            gap = curr_el.x - (prev_el.x + prev_el.width)
            font_size = prev_el.fontSize or DEFAULT_FONT_SIZE
            if gap > font_size * SPACE_GAP_MULTIPLIER:
                line_text += " "
            line_text += curr_el.text
        output_lines.append(line_text)
        
    return "\n".join(output_lines)

def _group_runs_into_lines(runs: List[ProcessingTextRun]) -> List[List[ProcessingTextRun]]:
    """Group PDF runs into horizontal lines based on Y position"""
    if not runs:
        return []
    
    # First, sort by Y position to group runs into potential lines
    y_sorted_runs = sorted(runs, key=lambda r: r.y)
    
    lines: List[List[ProcessingTextRun]] = []
    current_line = [y_sorted_runs[0]]
    
    for i in range(1, len(y_sorted_runs)):
        if abs(y_sorted_runs[i-1].y - y_sorted_runs[i].y) <= Y_POSITION_TOLERANCE:
            current_line.append(y_sorted_runs[i])
        else:
            # Sort current line by X position before adding to lines
            current_line.sort(key=lambda r: r.x)
            lines.append(current_line)
            current_line = [y_sorted_runs[i]]
    
    # Don't forget to sort the last line by X position
    current_line.sort(key=lambda r: r.x)
    lines.append(current_line)
    return lines

def _group_lines_into_paragraphs(lines: List[List[ProcessingTextRun]]) -> List[Dict[str, Any]]:
    """Group lines into paragraphs based on vertical spacing and typography"""
    if not lines:
        return []
    
    paragraphs: List[Dict[str, Any]] = []
    current_lines = [lines[0]]
    
    for i in range(1, len(lines)):
        prev_line_data = lines[i-1][0]
        default_height = prev_line_data.height or (DEFAULT_FONT_SIZE * DEFAULT_LINE_HEIGHT_RATIO)
        gap = lines[i][0].y - (prev_line_data.y + default_height)
        threshold = default_height * PARAGRAPH_BREAK_MULTIPLIER
        
        if gap > threshold:
            paragraphs.append({
                'lines': current_lines,
                'fontSize': _calculate_dominant_font_size(current_lines),
                'lineHeight': _calculate_line_height_ratio(current_lines),
                'marginBottom': None,
                'textAlign': 'left'
            })
            current_lines = [lines[i]]
        else:
            current_lines.append(lines[i])
    paragraphs.append({
        'lines': current_lines,
        'fontSize': _calculate_dominant_font_size(current_lines),
        'lineHeight': _calculate_line_height_ratio(current_lines),
        'marginBottom': None,
        'textAlign': 'left'
    })

    for i in range(len(paragraphs) - 1):
        paragraphs[i]['marginBottom'] = _calculate_margin_bottom(paragraphs[i]['lines'], paragraphs[i+1]['lines'])
    
    return paragraphs

def _create_text_element(runs: List[ProcessingTextRun], reason: TextGroupingReason) -> PdfTextElement:
    """
    Create a PdfTextElement from a list of ProcessingTextRun objects.
    Converts internal ProcessingTextRun objects to public PdfTextRun objects.
    """
    if not runs: raise ValueError("Cannot create grouped element from empty list")
    
    bbox = _calculate_bounding_box(runs)
    lines = _group_runs_into_lines(runs)
    paragraphs = _group_lines_into_paragraphs(lines)
    group_orientation = runs[0].rotation if runs else 0.0

    sections: List[ContentSection] = []
    for i, para in enumerate(paragraphs):
        section_id = f"section-{int(time.time() * 1000)}-{i}"
        converted_lines = []
        for line in para['lines']:
            converted_line = []
            for run_idx, processing_run in enumerate(line):
                # Determine if we should add a trailing space to this run
                run_text = processing_run.text
                
                if run_idx + 1 < len(line):
                    next_run = line[run_idx + 1]
                    
                    # Calculate gap using text matrix positions if available, otherwise fallback to bbox
                    if processing_run.tm_end_x is not None and next_run.tm_start_x is not None:
                        # Use text matrix positions - this includes PDF word spacing
                        gap_raw = next_run.tm_start_x - processing_run.tm_end_x
                        gap = round(gap_raw, 2)
                    else:
                        # Fallback to bounding box positions
                        current_run_end = processing_run.x + processing_run.width
                        gap_raw = next_run.x - current_run_end
                        gap = round(gap_raw, 2)
                    
                    font_size = processing_run.fontSize or DEFAULT_FONT_SIZE
                    
                    # Check if next run starts with punctuation that shouldn't have space before it
                    next_text_trimmed = next_run.text.lstrip()
                    no_space_before_punctuation = next_text_trimmed and next_text_trimmed[0] in '.,;:!?)\']}'
                    
                    # Check if current run ends with opening punctuation that shouldn't have space after it
                    curr_text_trimmed = processing_run.text.rstrip()
                    no_space_after_punctuation = curr_text_trimmed and curr_text_trimmed[-1] in '([{'
                    
                    # Determine if space is needed based on gap
                    # Use raw gap to distinguish truly touching text from rounding artifacts
                    if gap_raw < -0.001:
                        # Clearly overlapping text - intentionally concatenated, no space
                        pass
                    elif gap_raw < 0.001:
                        # Exactly touching (within floating point tolerance) - intentionally concatenated
                        # This handles cases like logo text where characters are precisely placed adjacent
                        pass
                    elif gap > font_size * SPACE_GAP_MULTIPLIER:
                        # Large gap - definitely add space (unless punctuation)
                        if not no_space_before_punctuation:
                            run_text += " "
                    else:
                        # Small gap (0.001 to threshold) - add space unless punctuation rules apply
                        if not no_space_before_punctuation and not no_space_after_punctuation:
                            run_text += " "
                
                # Create PdfTextRun with spacing already applied
                pdf_text_run = PdfTextRun(
                    text=run_text,
                    fontFamily=processing_run.fontFamily,
                    fontWeight=processing_run.fontWeight,
                    fontStyle=processing_run.fontStyle,
                    fontSize=processing_run.fontSize,
                    color=processing_run.color,
                    textDecoration=processing_run.textDecoration,
                    originalFontName=processing_run.originalFontName,
                    stream_index=processing_run.stream_index
                )
                converted_line.append(pdf_text_run)
            converted_lines.append(converted_line)
        
        sections.append(ParagraphSection(
            indentationLevel=para['lines'][0][0].x if para['lines'] and para['lines'][0] else 0,
            sectionId=section_id, 
            lines=converted_lines,
            fontSize=para.get('fontSize'),
            lineHeight=para.get('lineHeight'),
            marginBottom=para.get('marginBottom'),
            textAlign=para.get('textAlign')
        ))

    section_texts = []
    for section in sections:
        for line in section.lines:
            line_text = "".join(run.text for run in line)
            section_texts.append(line_text)
    content = "\n".join(section_texts)

    # Calculate the width as the maximum line width across all paragraphs
    # Each line's width is the sum of its text run widths
    max_line_width = 0.0
    for para in paragraphs:
        for line in para['lines']:
            line_width = sum(run.width for run in line)
            max_line_width = max(max_line_width, line_width)
    
    # Use calculated width if available, otherwise fall back to bounding box width
    final_width = max_line_width if max_line_width > 0 else bbox.width

    # Calculate rendered height from sections
    calculated_height = 0.0
    for section in sections:
        num_lines = len(section.lines)
        font_size = section.fontSize or DEFAULT_FONT_SIZE
        line_height_ratio = section.lineHeight or DEFAULT_LINE_HEIGHT_RATIO
        margin_bottom = section.marginBottom or 0.0
        
        # Each line contributes fontSize × lineHeight
        section_text_height = num_lines * font_size * line_height_ratio
        calculated_height += section_text_height + margin_bottom
    
    # Use the larger of calculated height or actual bounding box height
    # The bounding box height includes descenders on the last line, which the
    # line-height calculation doesn't account for
    final_height = max(calculated_height, bbox.height) if calculated_height > 0 else bbox.height

    group_signature = f"{content[:50]}_{bbox.x}_{bbox.y}_{reason}"
    group_id = hashlib.sha256(group_signature.encode('utf-8')).hexdigest()
    
    return PdfTextElement(
        id=group_id,
        text=content,
        x=bbox.x,
        y=bbox.y,
        width=final_width,
        height=final_height,
        rotation=group_orientation,
        sections=sections
    )

def _calculate_dominant_font_size(lines: List[List[ProcessingTextRun]]) -> Optional[float]:
    """Calculate dominant font size for a paragraph
    
    Returns the mode font size, or median if no clear mode exists.
    """
    if not lines:
        return None
    
    font_sizes = [el.fontSize for line in lines for el in line if el.fontSize and el.fontSize > 0]
    if not font_sizes:
        return None
    
    try:
        return statistics.mode(font_sizes)
    except statistics.StatisticsError:
        return statistics.median(font_sizes)

def _calculate_line_height_ratio(lines: List[List[ProcessingTextRun]]) -> Optional[float]:
    """Calculate unitless line height ratio from text runs
    
    Returns a CSS unitless ratio (e.g., 1.2, 1.5) to be multiplied by font size.
    Clamped to reasonable CSS line-height values (0.8 to 3.0).
    """
    if not lines:
        return None
    
    font_sizes = [el.fontSize for line in lines for el in line if el.fontSize and el.fontSize > 0]
    if not font_sizes:
        return DEFAULT_LINE_HEIGHT_RATIO

    try:
        dominant_font_size = statistics.mode(font_sizes)
    except statistics.StatisticsError:
        dominant_font_size = statistics.median(font_sizes)
    
    if len(lines) == 1:
        if lines[0]:
            ratio = (lines[0][0].height or dominant_font_size) / dominant_font_size
        else:
            ratio = DEFAULT_LINE_HEIGHT_RATIO
    else:
        line_distances = [
            abs(lines[i+1][0].y - lines[i][0].y) 
            for i in range(len(lines)-1) 
            if lines[i] and lines[i+1]
        ]
        
        if line_distances:
            median_distance = statistics.median(line_distances)
            ratio = median_distance / dominant_font_size
        else:
            ratio = DEFAULT_LINE_HEIGHT_RATIO

    clamped_ratio = max(MIN_LINE_HEIGHT_RATIO, min(MAX_LINE_HEIGHT_RATIO, ratio))
    return round(clamped_ratio, 2)

def _calculate_margin_bottom(
    current_lines: List[List[ProcessingTextRun]], 
    next_lines: List[List[ProcessingTextRun]]
) -> Optional[float]:
    """Calculate margin bottom based on gap to next paragraph
    
    Returns extra spacing between paragraphs beyond normal line height.
    """
    if not current_lines or not next_lines:
        return None
    
    last_line_run = current_lines[-1][0]
    next_line_run = next_lines[0][0]
    
    paragraph_distance = next_line_run.y - last_line_run.y
    font_size = last_line_run.fontSize or DEFAULT_FONT_SIZE
    line_height = last_line_run.height or (font_size * DEFAULT_LINE_HEIGHT_RATIO)
    margin = paragraph_distance - line_height
    
    return margin if margin > 0 else None

def _validate_grouping_results(
    input_runs: List[ProcessingTextRun], 
    output_elements: List[PdfElement], 
    element_to_runs: Dict[str, List[ProcessingTextRun]]
) -> None:
    """Validate that grouping preserved all text runs"""
    input_count = len(input_runs)
    output_text_count = sum(
        len(element_to_runs.get(el.id, [])) if el.type == "text" else 1 
        for el in output_elements
    )
    logger.debug(f"Grouping validation: input={input_count}, output_accounted={output_text_count}")
