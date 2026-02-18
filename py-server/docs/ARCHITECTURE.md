# Architecture Decision Records (ADR)

---

## Overview

This document records the key architectural decisions made during the refactoring of the PDF processing server. Each decision is documented with its context, rationale, alternatives considered, and consequences.

---

## ADR-001: Composition Over Inheritance

### Context

The original architecture had utility classes (`PdfContentEngine`, `ImageExtractionDevice`, `PDFPageModifier`) that were directly instantiated by extractor functions. We needed to decide how to structure the relationship between the new `PDFEngine` and its processors.

### Decision

Use composition: `PDFEngine` will compose `TextProcessor`, `ImageProcessor`, and `ContentModifier` as separate objects rather than inheriting from a base class or having processors inherit from the engine.

### Rationale

1. **Flexibility**: Processors can be replaced, mocked, or extended independently
2. **Testability**: Each processor can be tested in isolation
3. **Single Responsibility**: Engine handles coordination; processors handle domain logic
4. **Loose Coupling**: Processors don't need to know about each other
5. **Runtime Configuration**: Processors can be enabled/disabled dynamically

### Alternatives Considered

**Alternative 1: Inheritance Hierarchy**
```python
class BaseEngine:
    ...

class TextEngine(BaseEngine):
    ...

class ImageEngine(BaseEngine):
    ...

class PDFEngine(TextEngine, ImageEngine):  # Multiple inheritance
    ...
```
**Rejected because:** Multiple inheritance complexity, tight coupling, "diamond problem"

**Alternative 2: Mixins**
```python
class TextMixin:
    ...

class ImageMixin:
    ...

class PDFEngine(TextMixin, ImageMixin, BaseEngine):
    ...
```
**Rejected because:** Still tight coupling, harder to test, unclear ownership

### Consequences

**Positive:**
- Clean separation of concerns
- Easy to add new processors (e.g., `AnnotationProcessor`)
- Better testability
- Clear interface boundaries

**Negative:**
- Slightly more boilerplate (processor initialization)
- Need to pass engine reference to processors
- More files to maintain

**Mitigation:**
- Use factory methods for processor creation
- Document processor interface clearly
- Leverage Python's duck typing for flexibility

---

## ADR-002: Dependency Injection for Processors

### Context

Processors need access to shared resources (PDF document, page cache, configuration). We needed to decide how processors obtain these resources.

### Decision

Use constructor injection: Processors receive a reference to the `PDFEngine` instance during initialization.

```python
class TextProcessor:
    def __init__(self, engine: PDFEngine):
        self.engine = engine
```

### Rationale

1. **Explicit Dependencies**: Clear what each processor needs
2. **Testability**: Easy to inject mock engines for testing
3. **Resource Sharing**: Natural access to engine's resources
4. **Loose Coupling**: Processors depend on interface, not implementation
5. **Lifecycle Management**: Engine controls when processors are created

### Alternatives Considered

**Alternative 1: Service Locator Pattern**
```python
class TextProcessor:
    def __init__(self):
        self.engine = ServiceLocator.get('engine')
```
**Rejected because:** Hidden dependencies, harder to test, global state

**Alternative 2: Pass Resources Explicitly**
```python
class TextProcessor:
    def extract(self, pdf_document, page_cache, config):
        ...
```
**Rejected because:** Verbose, repetitive, tight coupling to resource types

**Alternative 3: Shared State Object**
```python
shared_state = SharedState(document, cache, config)
processor = TextProcessor(shared_state)
```
**Rejected because:** Less clear than passing engine, more indirection

### Consequences

**Positive:**
- Clear dependencies
- Easy to test with mocks
- Natural resource access pattern
- Simple to understand

**Negative:**
- Circular reference (engine → processor → engine)
- Processors coupled to engine interface

**Mitigation:**
- Use weak references if memory is a concern (not needed in practice)
- Define clear `PDFEngine` interface that processors depend on
- Document the dependency relationship

---

## ADR-003: Adapter Pattern for Backward Compatibility

### Context

Existing API endpoints and external code depend on the current function signatures in `extractors/`. We need to refactor internal implementation without breaking these contracts.

### Decision

Use the Adapter Pattern: Keep existing public function signatures but change implementation to delegate to new `PDFEngine`.

```python
# extractors/text_extractor.py (adapter)
def extract_text(file_path: str, start_page: int = 1, ...):
    """Public API maintained for backward compatibility."""
    with PDFEngine(file_path) as engine:
        return engine.extract_text(...)
```

### Rationale

1. **Zero Breaking Changes**: Existing code continues to work
2. **Gradual Migration**: Can refactor internals without touching API
3. **Rollback Safety**: Easy to revert to old implementation
4. **Clean Separation**: Public API separated from internal implementation
5. **Testing**: Can test both old and new implementations side-by-side

### Alternatives Considered

**Alternative 1: Version the API**
```python
# New API
from extractors.v2 import extract_text as extract_text_v2

# Old API
from extractors import extract_text  # Still works
```
**Rejected because:** Confusing for users, duplicates documentation, maintenance burden

**Alternative 2: Break Compatibility, Update All Callers**
```python
# New required API
engine = PDFEngine(file_path)
result = engine.extract_text(...)
```
**Rejected because:** High risk, breaks external code, lengthy migration

**Alternative 3: Feature Flags**
```python
def extract_text(use_new_engine=False):
    if use_new_engine:
        return new_implementation()
    else:
        return old_implementation()
```
**Rejected because:** Maintains two code paths, doubles testing burden

### Consequences

**Positive:**
- No breaking changes
- Low risk migration
- Easy rollback
- Existing tests continue to work

**Negative:**
- Temporary duplication (adapter + new code)
- Small performance overhead (negligible)
- Need to maintain adapter during transition

**Mitigation:**
- Remove adapters once migration is complete (Phase 6)
- Document that functions are adapters
- Add deprecation warnings (optional, for future major version)

---

## ADR-004: Separate processors/ Directory

### Context

Processing strategies (text_normalizer, text_grouping, position_tracking) were in `utils/` alongside pure utility functions (font_mapping, pdf_transforms). We needed to decide how to organize these different types of code.

### Decision

Create a separate `processors/` directory for processing strategies and algorithms, keeping `utils/` for pure functions only.

```
processors/
├── position_tracking.py     # Strategies (stateful algorithms)
├── text_normalizer.py       # Processing logic
├── text_grouping.py         # Semantic algorithms
└── graphics_state.py        # State management

utils/
├── font_mapping.py          # Pure utility functions
├── pdf_transforms.py        # Math utilities
└── validation.py            # Validation functions
```

### Rationale

1. **Clear Distinction**: Strategies vs. utilities have different characteristics
2. **Single Responsibility**: Each directory has one clear purpose
3. **Discoverability**: Easy to find processing strategies
4. **Growth Pattern**: Processors directory can grow with new algorithms
5. **Testing Strategy**: Different testing approaches for strategies vs. utils

### Alternatives Considered

**Alternative 1: Keep Everything in utils/**
```
utils/
├── position_tracking.py     # Strategies
├── text_normalizer.py       # Processing
├── font_mapping.py          # Utilities
└── pdf_transforms.py        # Utilities
```
**Rejected because:** Mixes concerns, harder to navigate, unclear purpose

**Alternative 2: Strategies Inside engine/**
```
engine/
├── pdf_engine.py
├── text_processor.py
├── strategies/
│   ├── position_tracking.py
│   └── text_grouping.py
```
**Rejected because:** Couples strategies to engine, less reusable

**Alternative 3: Algorithms in Each Processor File**
```
engine/
├── text_processor.py        # Contains all text algorithms
└── image_processor.py       # Contains all image algorithms
```
**Rejected because:** Large files, harder to test, poor separation

### Consequences

**Positive:**
- Clear code organization
- Easy to find processing strategies
- Strategies can be reused across processors
- Better separation of concerns

**Negative:**
- One more directory to navigate
- Need to update imports

**Mitigation:**
- Clear documentation of what goes where
- Good naming conventions
- Update import statements systematically

---

## ADR-005: Gradual Migration Strategy

### Context

We need to refactor a working production system without causing downtime or breaking existing functionality. We needed to decide on the migration approach.

### Decision

Use a gradual, phase-based migration where new code is added alongside old code, validated, then old code is removed only after the new code is proven stable.

### Rationale

1. **Risk Mitigation**: Can rollback at any phase
2. **Continuous Testing**: Old and new code can be tested in parallel
3. **No Downtime**: System remains functional throughout
4. **Incremental Progress**: Can pause/resume between phases
5. **Confidence Building**: Each phase builds on validated previous phase

### Alternatives Considered

**Alternative 1: Big Bang Rewrite**
- Rewrite everything at once
- Switch over in one deploy
**Rejected because:** High risk, no rollback, long testing cycle

**Alternative 2: Branch-Based Development**
- Develop new architecture in separate branch
- Merge when complete
**Rejected because:** Diverging codebases, merge conflicts, no parallel testing

**Alternative 3: Feature Flag Everything**
- All code has feature flags to toggle old/new
**Rejected because:** Complex configuration, doubles code paths, testing burden

### Consequences

**Positive:**
- Very low risk
- Easy rollback at any point
- Continuous validation
- Can spread work over time
- Production stays stable

**Negative:**
- Temporarily increased code size (old + new)
- Some code duplication during transition
- Longer total timeline

**Mitigation:**
- Clear phase boundaries
- Remove old code promptly after validation
- Document what's deprecated
- Automated tests for both paths

---

## ADR-006: Context Manager for Resource Lifecycle

### Context

PDF processing requires careful resource management (file handles, memory, temporary files). We needed to decide how to ensure resources are always properly cleaned up.

### Decision

Implement `PDFEngine` as a context manager (`__enter__`/`__exit__`) to guarantee resource cleanup even if exceptions occur.

```python
with PDFEngine(file_path) as engine:
    result = engine.extract_text(...)
# Resources automatically cleaned up here
```

### Rationale

1. **Guaranteed Cleanup**: Resources freed even on exception
2. **Pythonic Pattern**: Familiar to Python developers
3. **Explicit Lifecycle**: Clear when resources are acquired/released
4. **Exception Safety**: No resource leaks on error
5. **Composable**: Can nest context managers

### Alternatives Considered

**Alternative 1: Manual Cleanup**
```python
engine = PDFEngine(file_path)
try:
    result = engine.extract_text(...)
finally:
    engine.close()
```
**Rejected because:** Error-prone, boilerplate, easy to forget

**Alternative 2: Destructor (__del__)**
```python
class PDFEngine:
    def __del__(self):
        self.cleanup()
```
**Rejected because:** Unreliable timing, not guaranteed to run, debugging issues

**Alternative 3: Explicit Close Method Only**
```python
engine = PDFEngine(file_path)
result = engine.extract_text(...)
engine.close()  # Hope caller remembers!
```
**Rejected because:** Easy to forget, no exception safety

### Consequences

**Positive:**
- Guaranteed resource cleanup
- Exception safe
- Clear lifecycle
- Pythonic pattern

**Negative:**
- Requires with-statement usage
- Extra indentation level

**Mitigation:**
- Document context manager usage clearly
- Provide good error messages if used incorrectly
- Make it the only supported pattern

---

## ADR-007: Configuration via Dataclasses

### Context

Current configuration is done via dictionaries passed as parameters. We needed a more structured approach for the new engine while maintaining backward compatibility.

### Decision

Use Python `@dataclass` decorators for configuration objects (`EngineConfig`, `TextOptions`, etc.) with `from_dict()` methods for backward compatibility.

```python
@dataclass
class TextOptions:
    enable_grouping: bool = True
    max_horizontal_gap: float = 5.0
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'TextOptions':
        return cls(**config)
```

### Rationale

1. **Type Safety**: IDE autocompletion and type checking
2. **Clear Defaults**: Explicit default values
3. **Validation**: Can add validation methods
4. **Documentation**: Self-documenting structure
5. **Backward Compatible**: Can convert from dicts

### Alternatives Considered

**Alternative 1: Keep Using Dicts**
```python
config = {
    'enable_grouping': True,
    'max_horizontal_gap': 5.0
}
```
**Rejected because:** No type safety, easy to typo keys, no IDE support

**Alternative 2: Pydantic Models**
```python
class TextOptions(BaseModel):
    enable_grouping: bool = True
```
**Rejected because:** Adds dependency, overkill for internal config, runtime overhead

**Alternative 3: Named Tuples**
```python
TextOptions = namedtuple('TextOptions', ['enable_grouping', 'max_horizontal_gap'])
```
**Rejected because:** Immutable (may want to modify), no defaults, verbose

### Consequences

**Positive:**
- Type safety
- IDE support
- Self-documenting
- Easy validation
- Backward compatible

**Negative:**
- Slightly more verbose than dicts
- Need conversion methods

**Mitigation:**
- Provide `from_dict()` for migration
- Good defaults minimize verbosity
- Document configuration clearly

## ADR-008: Dual-Pipeline Extraction & Hydration

### Context
We have two conflicting requirements:
* **Semantic Structure:** We need accurate text flow, reading order, and layout analysis (Best provided by `PDFMiner`).
* **Visual Fidelity:** We need precise vector paths, gradients, and correct rendering of complex graphics (Best provided by `PikePDF`).

No single library provides both at the quality level required.

### Decision
Implement a **Dual-Pipeline Architecture**:
1.  **Stream Pipeline (`PDFMiner`):** Extracts text and creates "Placeholder" vectors based on the paint stream order.
2.  **Rich Pipeline (`PikePDF`):** Extracts high-fidelity vector objects with full path data, styling, and gradients.
3.  **Hydration Step:** Merge the two pipelines by matching Rich Vectors to Stream Placeholders based on spatial proximity.

### Rationale
* **Best of Both Worlds:** We get the semantic ordering of PDFMiner and the visual quality of PikePDF.
* **Orphan Handling:** The hydration process identifies vectors that don't fit the stream flow (orphans), allowing us to handle them explicitly (e.g., background layers).

### Consequences
* **Complexity:** Requires maintaining two extraction logical paths.
* **Coordinate Synchronization:** Critical requirement that both pipelines normalize coordinates (MediaBox) exactly the same way, or hydration fails (see Implementation Notes).

---

## ADR-009: Deferred Vector Merging

### Context
When grouping vectors (e.g., for a logo), we initially merged them into SVG strings immediately upon detection. This destroyed the `stream_index` metadata required to determine the Z-order (layering) relative to other elements.

### Decision
Adopt a **"Group First, Merge Last"** pattern:
1.  **Group:** Collect vectors into lists but keep them as object instances.
2.  **Process:** Perform overlap detection and sorting on the object lists using their metadata.
3.  **Merge:** Only convert to SVG strings in the final step, ensuring the merged object inherits the correct `stream_index`.

### Rationale
* **Z-Order Correctness:** Allows sorting complex, overlapping groups (e.g., a sun behind trees) based on the paint order of their constituent parts.
* **Data Preservation:** Metadata is preserved until the final render decision is made.
* **Sort Stability:** We explicitly handle `None` values in the sort key to prevent indeterminate ordering during the merge phase:
    ```python
    key=lambda v: v.stream_index if v.stream_index is not None else float('inf')
    ```