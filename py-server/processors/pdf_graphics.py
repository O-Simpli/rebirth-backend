import logging
from typing import Tuple
import numpy as np
from pikepdf import Name

from constants.pdf_operators import (
    OP_SAVE_STATE, OP_RESTORE_STATE, OP_CTM,
    OP_SET_RGB_COLOR_FILL, OP_BEGIN_TEXT, OP_END_TEXT,
    OP_MOVE_TEXT, OP_MOVE_TEXT_SET_LEADING, OP_SET_TEXT_MATRIX, OP_NEXT_LINE,
    OP_SET_FONT, OP_SET_CHAR_SPACING, OP_SET_WORD_SPACING,
    OP_SET_HORIZ_SCALING, OP_SET_LEADING, OP_SET_TEXT_RISE
)

logger = logging.getLogger(__name__)

def normalize_operator(operator) -> bytes:
    op_name = operator.operator
    if isinstance(op_name, str):
        return op_name.encode('latin-1')
    elif isinstance(op_name, bytes):
        return op_name
    else:
        try:
            return str(op_name).encode('latin-1')
        except:
            return b''

class GraphicsStateTracker:
    def __init__(self, page_height: float = 792.0):
        self.page_height = page_height
        self.ctm = np.identity(3, dtype=float)
        self.state_stack = []
        self.text_matrix = np.identity(3, dtype=float)
        self.text_line_matrix = np.identity(3, dtype=float)
        self.font_size = 12.0
        self.font_resource_name: Name = None
        self.character_spacing = 0.0
        self.word_spacing = 0.0
        self.horizontal_scaling = 100.0
        self.leading = 0.0
        self.text_rise = 0.0
        self.in_text_object = False
        self.non_stroking_color = (0, 0, 0)

    def save_state(self):
        state = {
            'ctm': self.ctm.copy(),
            'text_matrix': self.text_matrix.copy(),
            'text_line_matrix': self.text_line_matrix.copy(),
            'font_size': self.font_size,
            'font_resource_name': self.font_resource_name,
            'leading': self.leading,
            'character_spacing': self.character_spacing,
            'word_spacing': self.word_spacing,
            'horizontal_scaling': self.horizontal_scaling,
            'text_rise': self.text_rise,
            'in_text_object': self.in_text_object,
        }
        self.state_stack.append(state)

    def restore_state(self):
        if self.state_stack:
            state = self.state_stack.pop()
            self.ctm = state['ctm']
            self.text_matrix = state['text_matrix']
            self.text_line_matrix = state['text_line_matrix']
            self.font_size = state['font_size']
            self.font_resource_name = state['font_resource_name']
            self.leading = state['leading']
            self.character_spacing = state['character_spacing']
            self.word_spacing = state['word_spacing']
            self.horizontal_scaling = state['horizontal_scaling']
            self.text_rise = state['text_rise']
            self.in_text_object = state['in_text_object']

    def update_ctm(self, a: float, b: float, c: float, d: float, e: float, f: float):
        new_matrix = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)
        self.ctm = np.dot(self.ctm, new_matrix)

    def get_text_transform(self) -> Tuple[float, float, float]:
        """
        Calculates the final text position (baseline-left) and rotation.
        
        Returns:
            A tuple containing (x, y, angle_degrees).
        """
        combined_matrix = np.dot(self.ctm, self.text_matrix)
        x = float(combined_matrix[0, 2])
        y = float(combined_matrix[1, 2])
        
        # Extract rotation angle from the combined matrix
        a = combined_matrix[0, 0]
        b = combined_matrix[1, 0]
        angle_rad = np.arctan2(b, a)
        angle_deg = np.degrees(angle_rad)
        
        return x, y, angle_deg

    def _update_graphics_state(self, operator) -> None:
        op_name_bytes = normalize_operator(operator)
        operands = operator.operands

        try:
            if op_name_bytes == OP_SAVE_STATE: 
                self.save_state()
            elif op_name_bytes == OP_RESTORE_STATE: 
                self.restore_state()
            elif op_name_bytes == OP_CTM and len(operands) == 6: 
                self.update_ctm(*[float(op) for op in operands])
            elif op_name_bytes == OP_SET_RGB_COLOR_FILL and len(operands) == 3:
                self.non_stroking_color = tuple(float(op) for op in operands)
            elif op_name_bytes == OP_BEGIN_TEXT:
                self.in_text_object = True
                self.text_matrix = np.identity(3, dtype=float)
                self.text_line_matrix = np.identity(3, dtype=float)
            elif op_name_bytes == OP_END_TEXT: 
                self.in_text_object = False
            elif op_name_bytes == OP_MOVE_TEXT and len(operands) == 2:
                tx, ty = float(operands[0]), float(operands[1])
                translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
                self.text_line_matrix = np.dot(self.text_line_matrix, translation)
                self.text_matrix = self.text_line_matrix.copy()
            elif op_name_bytes == OP_MOVE_TEXT_SET_LEADING and len(operands) == 2:
                tx, ty = float(operands[0]), float(operands[1])
                self.leading = -ty
                translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
                self.text_line_matrix = np.dot(self.text_line_matrix, translation)
                self.text_matrix = self.text_line_matrix.copy()
            elif op_name_bytes == OP_SET_TEXT_MATRIX and len(operands) == 6:
                a, b, c, d, e, f = [float(op) for op in operands]
                self.text_matrix = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)
                self.text_line_matrix = self.text_matrix.copy()
            elif op_name_bytes == OP_NEXT_LINE:
                translation = np.array([[1, 0, 0], [0, 1, -self.leading], [0, 0, 1]], dtype=float)
                self.text_line_matrix = np.dot(self.text_line_matrix, translation)
                self.text_matrix = self.text_line_matrix.copy()
            elif op_name_bytes == OP_SET_FONT and len(operands) >= 2:
                self.font_resource_name = operands[0]
                self.font_size = float(operands[1])
            elif op_name_bytes == OP_SET_CHAR_SPACING and len(operands) >= 1: 
                self.character_spacing = float(operands[0])
            elif op_name_bytes == OP_SET_WORD_SPACING and len(operands) >= 1: 
                self.word_spacing = float(operands[0])
            elif op_name_bytes == OP_SET_HORIZ_SCALING and len(operands) >= 1: 
                self.horizontal_scaling = float(operands[0])
            elif op_name_bytes == OP_SET_LEADING and len(operands) >= 1: 
                self.leading = float(operands[0])
            elif op_name_bytes == OP_SET_TEXT_RISE and len(operands) >= 1: 
                self.text_rise = float(operands[0])

        except (ValueError, IndexError) as e:
            logger.warning(f"Error updating graphics state for operator {op_name_bytes}: {e}")