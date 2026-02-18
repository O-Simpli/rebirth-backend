"""
Position tracking strategies for PDF text extraction.

Different text scenarios require different position tracking logic:
- Regular text: handles font subsets, drop caps, table columns
- Normalized rotated text: sequential style changes at same position
- Future: bidirectional, vertical, complex scripts
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

# Position jump threshold: forward/backward jumps larger than 2x font size indicate explicit positioning
FONT_SIZE_MULTIPLIER_FORWARD_JUMP = 2.0
POSITION_BACKWARDS_TOLERANCE = 1.0


class PositionTrackingStrategy(ABC):
    """Base strategy for determining when to use tracked vs incoming text position."""
    
    @abstractmethod
    def should_use_tracked_position(
        self,
        incoming_x: float,
        last_x: float,
        font_size: float,
        same_line: bool,
        is_backwards: bool
    ) -> bool:
        """
        Determine if tracked position should be used for character placement.
        
        Args:
            incoming_x: X position from PDF text matrix
            last_x: X position from previous character advancement
            font_size: Current font size
            same_line: Whether on same Y position as previous text
            is_backwards: Whether movement indicates faux-bold duplicate
            
        Returns:
            True to use tracked position, False to use incoming position
        """
        pass


class RegularTextStrategy(PositionTrackingStrategy):
    """
    Position tracking for regular PDF content.
    
    Handles three scenarios:
    1. Font subsets: consecutive operators for same word use tracked position
    2. Drop caps/table columns: large position jumps use incoming position
    3. Faux-bold duplicates: already filtered by is_backwards check
    """
    
    def should_use_tracked_position(
        self,
        incoming_x: float,
        last_x: float,
        font_size: float,
        same_line: bool,
        is_backwards: bool
    ) -> bool:
        if not same_line or is_backwards:
            return False
        
        horizontal_jump = incoming_x - last_x
        
        # Large jumps (forward or backward) indicate explicit positioning
        threshold = font_size * FONT_SIZE_MULTIPLIER_FORWARD_JUMP
        has_significant_jump = abs(horizontal_jump) > threshold
        
        # Use tracked position only for sequential same-line text without large jumps
        return not has_significant_jump


class NormalizedRotatedStrategy(PositionTrackingStrategy):
    """
    Position tracking for normalized rotated text.
    
    When rotated text is physically rotated to horizontal for extraction,
    different style runs (bold→regular→italic) appear at nearly identical
    positions. Must use tracked position to advance sequentially.
    
    Disables backwards jump detection that would break sequential advancement.
    """
    
    def should_use_tracked_position(
        self,
        incoming_x: float,
        last_x: float,
        font_size: float,
        same_line: bool,
        is_backwards: bool
    ) -> bool:
        if not same_line or is_backwards:
            return False
        
        horizontal_jump = incoming_x - last_x
        
        # Only detect forward jumps (tables, columns)
        # Ignore backwards jumps to allow sequential advancement through style changes
        has_forward_jump = horizontal_jump > font_size * FONT_SIZE_MULTIPLIER_FORWARD_JUMP
        
        return not has_forward_jump
