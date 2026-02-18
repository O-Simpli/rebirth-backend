# Documentation Library

This directory contains comprehensive documentation for the PDF processing server, focusing on architectural patterns, implementation details, and non-obvious solutions.

## Core Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md)
Architectural decision records (ADRs) documenting key design patterns and their rationale:
- Composition over inheritance
- Dependency injection for processors
- Adapter pattern for backward compatibility
- Resource lifecycle management with context managers
- Configuration via dataclasses

### [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
Critical implementation details that are non-obvious and essential for maintaining correctness:
- MediaBox coordinate normalization
- Stroke width padding for thin lines
- Form XObject vector processing
- Text matrix advancement tracking
- Known edge cases and workarounds

## Contributing

When adding new features or fixing bugs:
1. Document architectural decisions in ARCHITECTURE.md using ADR format
2. Document non-obvious implementations in IMPLEMENTATION_NOTES.md
3. Include code references with file paths and line numbers
4. Explain **what** was done, **why** it was necessary, and **how** it works
