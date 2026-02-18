"""
PDF Content Normalization Module

Performs low-level PDF stream manipulation to normalize rotated text. It parses
a page's content stream, identifies rotated text elements, and creates a new,
clean PDF page with that text rendered horizontally and marked with its
original rotation data.
"""

import io
import math
import logging
import re
import numpy as np
from typing import Optional, Tuple

from pikepdf import Pdf, Page, String, Name, Dictionary, parse_content_stream, unparse_content_stream
from processors.pdf_graphics import GraphicsStateTracker, normalize_operator
from constants.pdf_operators import TEXT_SHOWING_OPS, OP_SET_TEXT_MATRIX, OP_SHOW_TEXT, OP_SHOW_TEXT_ARRAY
from utils.pdf_transforms import decompose_ctm, get_text_pivot_point

logger = logging.getLogger(__name__)

ORIENTATION_MARKER_FORMAT = "<|ORI:{angle:.2f}>"
ORIENTATION_REGEX = re.compile(r"<\|ORI:(?P<angle>-?\d+(?:\.\d*)?)")

def normalize_rotated_text_on_page(
    original_page: Page, 
    page_num: int, 
    rotation_threshold: float = 1.0
) -> Optional[bytes]:
    """
    Creates a new PDF page containing only the rotated text from the given
    Page object, rendered horizontally and marked with its original rotation.
    """
    logger.debug(f"Starting normalization for page {page_num}.")
    
    rotated_text_found = False
    
    try:
        try:
            media_box = original_page['/MediaBox']
            original_page_size = (float(media_box[2]), float(media_box[3]))
            gs_page_height = float(media_box[3])
        except Exception as e:
            logger.warning(f"Could not get page size for page {page_num}: {e}. Falling back to A4.")
            original_page_size = (595, 842) # Default A4
            gs_page_height = 842.0

        gs = GraphicsStateTracker(page_height=gs_page_height)
        operators = parse_content_stream(original_page)

        rotated_segments = []
        is_building_rotated_line = False
        current_line_ops = []
        current_line_angle = 0.0
        current_line_y_pos = 0.0
        last_matrix = None
        last_font_size = 0.0

        for op in operators:
            # Track when a text object from the original PDF ends to finalize our line
            was_in_text_object = gs.in_text_object
            gs._update_graphics_state(op)
            is_now_in_text_object = gs.in_text_object

            # If we were building a line and the original text object ends (e.g., ET operator), finalize our line.
            if is_building_rotated_line and was_in_text_object and not is_now_in_text_object:
                current_line_ops.append(([], b'ET'))
                rotated_segments.append({
                    'y_pos': current_line_y_pos,
                    'ops': current_line_ops.copy()
                })

                # Reset state for the next potential rotated line
                is_building_rotated_line = False
                current_line_ops = []
                last_matrix = None

            op_name_bytes = normalize_operator(op)
            if op_name_bytes not in TEXT_SHOWING_OPS:
                continue

            tm, ctm = gs.text_matrix, gs.ctm
            combined_matrix = np.dot(ctm, tm)

            # Extract the 6-element list from the 3x3 numpy matrix
            a, b = combined_matrix[0, 0], combined_matrix[1, 0]
            c, d = combined_matrix[0, 1], combined_matrix[1, 1]
            e, f = combined_matrix[0, 2], combined_matrix[1, 2]
            
            transformation = decompose_ctm([a, b, c, d, e, f])
            angle = transformation.rotation

            if abs(angle) >= rotation_threshold:
                if not op.operands:
                    continue

                rotated_text_found = True

                # Check if the current text chunk continues the previous line or is part of same paragraph
                is_on_same_line = False
                is_same_paragraph = False
                if is_building_rotated_line and last_matrix is not None:
                    y_diff = abs(combined_matrix[1, 2] - last_matrix[1, 2])
                    angle_diff = abs(angle - current_line_angle)

                    # Tolerance for same line based on previous font size
                    y_tolerance = last_font_size * 0.5
                    
                    # Tolerance for same paragraph - allow for line spacing
                    paragraph_y_tolerance = last_font_size * 2.0

                    if y_diff <= y_tolerance and angle_diff < 1.0:
                        is_on_same_line = True
                    elif y_diff <= paragraph_y_tolerance and angle_diff < 1.0:
                        is_same_paragraph = True

                # If we were building a line but this chunk is not part of the same line or paragraph, finalize the old one.
                if is_building_rotated_line and not is_on_same_line and not is_same_paragraph:
                    current_line_ops.append(([], b'ET'))
                    rotated_segments.append({
                        'y_pos': current_line_y_pos,
                        'ops': current_line_ops.copy()
                    })

                    # Reset state to start a new line
                    is_building_rotated_line = False
                    current_line_ops = []
                    last_matrix = None

                # This is the first segment of a new rotated line or continuing paragraph.
                if not is_building_rotated_line:
                    is_building_rotated_line = True
                    current_line_angle = angle # Store angle for the final marker
                    current_line_y_pos = combined_matrix[1, 2]  # Store Y position for sorting

                    font_size = gs.font_size
                    pos_x, pos_y = combined_matrix[0, 2], combined_matrix[1, 2]

                    adjusted_pos_x, adjusted_pos_y = get_text_pivot_point(
                        pos_x, pos_y, gs.font_size, angle
                    )

                    new_tm_tuple = ([1, 0, 0, 1, adjusted_pos_x, adjusted_pos_y], b'Tm')
                    current_color = gs.non_stroking_color
                    color_op_tuple = (list(current_color), b'rg')

                    current_line_ops.append(([], b'BT'))
                    current_line_ops.append(color_op_tuple)
                    current_line_ops.append(new_tm_tuple)

                # If this is a new line in the same paragraph, move to next line position
                elif is_same_paragraph:
                    # Move to next line by adjusting Y position
                    font_size = gs.font_size
                    line_spacing = font_size  # Standard line spacing

                    # Get the first line's position for consistent X alignment
                    first_tm_op = None
                    for op_tuple in current_line_ops:
                        if op_tuple[1] == OP_SET_TEXT_MATRIX:
                            first_tm_op = op_tuple
                            break
                    
                    if first_tm_op:
                        # Use the first line's X coordinate for alignment
                        first_x = first_tm_op[0][4]
                    
                        # Get the last position from current ops
                        last_tm_op = None
                        for i in range(len(current_line_ops) - 1, -1, -1):
                            if current_line_ops[i][1] == OP_SET_TEXT_MATRIX:
                                last_tm_op = current_line_ops[i]
                                break
                        
                        if last_tm_op:
                            last_y = last_tm_op[0][5]  # Y coordinate from Tm matrix
                            new_y = last_y - line_spacing
                            new_line_tm = ([1, 0, 0, 1, first_x, new_y], OP_SET_TEXT_MATRIX)
                            current_line_ops.append(new_line_tm)

                # Run all line segments
                text_operand = op.operands[0]
                final_text_operand = None
                if op_name_bytes == OP_SHOW_TEXT and isinstance(text_operand, String):
                    final_text_operand = text_operand
                elif op_name_bytes == OP_SHOW_TEXT_ARRAY and isinstance(text_operand, list):
                    final_text_operand = String(b''.join([bytes(p) for p in text_operand if isinstance(p, String)]))

                if not (final_text_operand and str(final_text_operand).strip()):
                    continue

                current_font_name = gs.font_resource_name or Name('/F1')
                visible_font_op = ([current_font_name, gs.font_size], b'Tf')
                visible_text_op = ([final_text_operand], b'Tj')

                # Create the operand for the invisible rotation marker and add it after each segment
                marker = ORIENTATION_MARKER_FORMAT.format(angle=current_line_angle)
                marker_font_name = Name('/MarkerFont')
                invisible_font_op = ([marker_font_name, 0.01], b'Tf')
                invisible_text_op = ([String(marker)], b'Tj')

                current_line_ops.extend([
                    visible_font_op,
                    visible_text_op,
                    invisible_font_op,
                    invisible_text_op
                ])

                # Update state for the next iteration's same-line check
                last_matrix = combined_matrix.copy()
                last_font_size = gs.font_size

            # If we encounter non-rotated text while building a rotated line, this marks the end.
            elif is_building_rotated_line:
                current_line_ops.append(([], b'ET'))
                rotated_segments.append({
                    'y_pos': current_line_y_pos,
                    'ops': current_line_ops.copy()
                })

                is_building_rotated_line = False
                current_line_ops = []
                last_matrix = None

        # After the loop, finalize any line that was still being built.
        if is_building_rotated_line:
            current_line_ops.append(([], b'ET'))
            rotated_segments.append({
                'y_pos': current_line_y_pos,
                'ops': current_line_ops.copy()
            })

        # Sort segments by Y position (descending for proper reading order)
        rotated_segments.sort(key=lambda x: x['y_pos'], reverse=True)
        
        # Build final operators list in correct order
        new_operators = []
        for segment in rotated_segments:
            new_operators.extend(segment['ops'])

        if not rotated_text_found:
            logger.debug(f"Page {page_num}: No rotated text found. Normalization not required.")
            return None
        
        new_pdf = Pdf.new()
        new_page = new_pdf.add_blank_page(page_size=original_page_size)

        if original_page.Resources and original_page.Resources.get('/Font'):
            new_resources = Dictionary()
            new_font_dict = Dictionary()
            original_font_dict = original_page.Resources.Font
            
            for font_name, font_obj in original_font_dict.items():
                new_font_dict[font_name] = new_pdf.copy_foreign(font_obj)
              
            new_font_dict[Name('/MarkerFont')] = Dictionary({
                '/Type': Name.Font,
                '/Subtype': Name.Type1,
                '/BaseFont': Name.Helvetica,
            })
              
            new_resources.Font = new_font_dict
            new_page.Resources = new_resources
            
        new_content_stream = unparse_content_stream(new_operators)
        new_page.Contents = new_pdf.make_stream(new_content_stream)

        buffer = io.BytesIO()
        new_pdf.save(buffer)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"An unexpected error occurred during normalization of page {page_num}: {e}", exc_info=True)
        return None

def parse_and_clean_run(text: str) -> Tuple[str, float]:
    """
    Checks for the rotation marker in a text string, parses it,
    and returns the cleaned text and the rotation angle.
    """
    rotation = 0.0
    match = ORIENTATION_REGEX.search(text)
    
    if match:
        data = match.groupdict()
        rotation = -round(float(data['angle']), 2) # PDF has inverted Y-axis
        return text[:match.start()], rotation
    
    return text, rotation