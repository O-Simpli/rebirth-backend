"""
Base processor protocol and abstract classes.

Defines the interface that all processors must implement for integration
with the PDFEngine. Uses Protocol for duck typing and ABC for enforcement.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from engine.pdf_engine import PDFEngine

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for all PDF processors.
    
    Processors receive a PDFEngine reference to access shared resources
    and must implement initialize() and cleanup() for resource management.
    """
    
    def __init__(self, engine: 'PDFEngine'):
        """
        Initialize processor with engine reference.
        
        Args:
            engine: PDFEngine instance that owns this processor
        """
        self.engine = engine
        self._initialized = False
        logger.debug(f"{self.__class__.__name__} created with engine reference")
    
    def initialize(self) -> None:
        """
        Initialize processor-specific resources.
        
        Called by the engine after processor creation but before use.
        Override this method to set up internal state, open connections,
        or perform one-time setup.
        
        This is separate from __init__ to allow controlled initialization
        timing by the engine.
        """
        if self._initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return
        
        self._initialized = True
        logger.debug(f"{self.__class__.__name__} initialized")
    
    def cleanup(self) -> None:
        """
        Clean up processor-specific resources.
        
        Called by the engine during shutdown or context manager exit.
        Override this method to release resources, close connections,
        or perform cleanup operations.
        
        This method should be idempotent (safe to call multiple times).
        """
        if not self._initialized:
            return
        
        self._initialized = False
        logger.debug(f"{self.__class__.__name__} cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        """Check if processor has been initialized."""
        return self._initialized
    
    def validate_state(self) -> bool:
        """
        Validate that processor is in a valid state for operations.
        
        Returns:
            True if processor is ready, False otherwise
        """
        if not self._initialized:
            logger.error(f"{self.__class__.__name__} not initialized")
            return False
        
        if self.engine is None:
            logger.error(f"{self.__class__.__name__} has no engine reference")
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}({status})"


class ProcessorProtocol(Protocol):
    """
    Protocol defining the interface for processors.
    
    This uses Python's structural subtyping (duck typing) to define what
    a processor must implement without requiring inheritance. This is more
    flexible than ABC for type checking.
    
    Any class implementing these methods can be used as a processor,
    even without inheriting from BaseProcessor.
    """
    
    engine: 'PDFEngine'
    _initialized: bool
    
    def initialize(self) -> None:
        """Initialize processor resources."""
        ...
    
    def cleanup(self) -> None:
        """Clean up processor resources."""
        ...
    
    def validate_state(self) -> bool:
        """Validate processor state."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check initialization status."""
        ...


class ProcessorRegistry:
    """
    Registry for managing processor instances.
    
    Provides centralized tracking of all processors attached to an engine,
    with support for initialization, cleanup, and lifecycle management.
    
    Used internally by PDFEngine to manage its processors.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._processors: dict[str, BaseProcessor] = {}
        self._initialization_order: list[str] = []
    
    def register(self, name: str, processor: BaseProcessor) -> None:
        """
        Register a processor.
        
        Args:
            name: Unique name for the processor (e.g., "text", "image")
            processor: Processor instance to register
        """
        if name in self._processors:
            logger.warning(f"Processor '{name}' already registered, replacing")
        
        self._processors[name] = processor
        if name not in self._initialization_order:
            self._initialization_order.append(name)
        
        logger.debug(f"Registered processor: {name}")
    
    def get(self, name: str) -> Optional[BaseProcessor]:
        """
        Get processor by name.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance or None if not found
        """
        return self._processors.get(name)
    
    def initialize_all(self) -> None:
        """Initialize all registered processors in registration order."""
        for name in self._initialization_order:
            processor = self._processors.get(name)
            if processor:
                try:
                    processor.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize processor '{name}': {e}")
                    raise
    
    def cleanup_all(self) -> None:
        """Clean up all processors in reverse registration order."""
        for name in reversed(self._initialization_order):
            processor = self._processors.get(name)
            if processor:
                try:
                    processor.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up processor '{name}': {e}")
                    # Continue cleanup despite errors
    
    def validate_all(self) -> bool:
        """
        Validate all processors are in valid state.
        
        Returns:
            True if all processors valid, False otherwise
        """
        all_valid = True
        for name, processor in self._processors.items():
            if not processor.validate_state():
                logger.error(f"Processor '{name}' in invalid state")
                all_valid = False
        
        return all_valid
    
    @property
    def processor_count(self) -> int:
        """Get number of registered processors."""
        return len(self._processors)
    
    @property
    def processor_names(self) -> list[str]:
        """Get list of registered processor names."""
        return list(self._processors.keys())
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ProcessorRegistry({self.processor_count} processors: {self.processor_names})"
