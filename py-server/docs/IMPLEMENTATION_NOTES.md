# Implementation Notes

This document captures critical implementation details that aren't obvious from the code alone. Each section explains the problem being solved, the solution approach, and why it matters.

---

## Coordinate Systems & Transformations

### MediaBox Origin Handling

PDFs don't always start at `(0,0)`. Professional design tools like Canva, Figma, and Adobe InDesign frequently produce PDFs with non-zero MediaBox origins—for example, `[0, 7.92, 612, 799.92]` instead of the standard `[0, 0, 612, 792]`.

This is intentional, not a bug:
* Preserves WYSIWYG accuracy between the editor and PDF output.
* Maintains print safety margins to prevent content cutoff.
* Supports template systems with consistent safe zones.
* Preserves web canvas coordinate systems from browser-based tools.

The PDF specification explicitly permits this. The origin refers to the page's corner, not the coordinate space itself.

**The Challenge:** We use two different libraries—`pdfminer` for text stream ordering and `pikepdf` for rich vector extraction. If they interpret the MediaBox origin differently, their coordinate systems drift apart. This causes extracted vectors to fail matching with their stream-order placeholders.

**The Symptom:** Vectors appear as "orphans" at the end of the rendering list instead of maintaining their correct z-index position. Large background shapes render on top of text and images, obscuring content.

**The Solution:** Both extraction systems apply coordinate normalization:
```python
y_screen = page_height - (y_pdf - mediabox_bottom)
```
This ensures that a vector at PDF coordinate `Y=654.72` in a document with MediaBox bottom=`7.92` maps to the same screen position regardless of which library extracted it.

**Components affected:** Text extraction, vector extraction, image positioning

### Image Unit Square Mapping

PDF images are defined on a normalized unit square from `(0,0)` to `(1,1)`. The Current Transformation Matrix (CTM) transforms this unit square to the image's actual position, size, rotation, and skew on the page.

To calculate an image's bounding box:
1.  Start with the four unit square corners: `[(0,0), (1,0), (1,1), (0,1)]`
2.  Apply the CTM to each corner.
3.  Convert from PDF Y-up coordinates to web Y-down coordinates.
4.  Find the bounding rectangle of the transformed corners.

This approach handles all transformations correctly—rotation, skew, reflection, and non-uniform scaling—without needing to decompose the transformation matrix.

**Example:**
```python
# CTM for 200×150 image at (100, 200), no rotation
ctm = [200, 0, 0, 150, 100, 200]

# Unit square (0,0) → (100, 200) - bottom-left
# Unit square (1,1) → (300, 350) - top-right
# Result: {x: 100, y: 200, width: 200, height: 150}
```

**Components affected:** Image extraction, vector extraction

---

## Vector Processing & Z-Order

### Vector Grouping & Z-Order Preservation

Complex vector graphics (like logos) are composed of hundreds of small paths. To ensure they render correctly and perform efficiently, we group related vectors together. The critical challenge is preserving the correct Z-order (visual layering) during this process.

**The "Early Merge" Fallacy:** A naive approach is to merge vectors into a single SVG as soon as they are grouped. This destroys the `stream_index` (PDF paint order) of the individual components. If these merged groups later need to be sorted against other elements (e.g., a background sun vs. foreground trees), the sorting fails because the original order information is lost.

**The Solution: "Group First, Merge Last" Architecture**

1.  **Group (Phase 1):** Collect vectors by `clippingRegionId` but keep them as raw lists (`List[PdfVectorElement]`). Do not create merged SVGs yet.
2.  **Detect Overlaps (Phase 2):** Use Union-Find to detect which vector groups spatially overlap.
    * *Super-Groups:* This logic calculates the transitive closure of overlaps. If Group A overlaps Group B, and Group B overlaps Group C, they are linked into a single "Super-Group" `[A, B, C]`. This is critical for complex, interlocking graphics (e.g., a logo behind a text box behind a ribbon), ensuring they are processed as a single unit rather than fragmented layers.
3.  **Flatten & Sort (Phase 3):** Flatten the overlapping groups into a single list of vectors and sort by `stream_index`.
    * *Sort Stability:* We use an explicit sort key: `key=lambda v: v.stream_index if v.stream_index is not None else float('inf')`. This forces any elements without a stream index to the back of the stack, preventing Python's sort from producing indeterminate results which would cause random Z-order flickering.
4.  **Merge Once (Phase 4):** Merge the sorted list into a single SVG. The final element inherits the *earliest* `stream_index` (e.g., 44), ensuring the entire composite graphic sits at the correct depth in the z-stack.

**Components affected:** Vector grouping, overlap detection

### Robust SVG Merging & Clipping

Merging raw SVGs from different vectors requires detailed handling to prevent content breakage.

**1. ID Collision Prevention (Namespacing)**
Vectors often reuse internal IDs (e.g., `id="clip_5"`). When merged, the browser applies the first definition it finds to all elements.
* **Fix:** Every ID (gradients, patterns, clipPaths) is namespaced with a unique vector index: `id="v0_clip_5"`.

**2. Coordinate Alignment (SVG Transforms)**
Vectors in the same group often reside at different positions, but they share a Clipping Path defined in absolute PDF coordinates.
* **The Trap:** Attempting to shift path data using Regex (`d="M 10..."`) is fragile and breaks on scientific notation or arcs.
* **The Fix:** Leave path data untouched. Apply `transform="translate(-v.x, -v.y)"` to the `<path>` inside the `<clipPath>`. This forces the browser to align the absolute clip path with the vector's local coordinate system.

**3. Structure Preservation (Wrapper Groups)**
* **Fix:** Do not extract `<path>` tags directly. Extract the entire inner body of the SVG and wrap it in a `<g transform="...">`. This preserves nested clips and transforms that would otherwise be lost.

**Components affected:** SVG generation, vector merging

### Stroke Width Padding

Stroked paths need their bounding boxes expanded by the stroke width on all sides. This isn't optional—it's critical for two reasons:

1.  **Zero-height paths:** A horizontal line from (0, 100) to (500, 100) has a mathematical bounding box with height=0. But visually, with a 1pt stroke, it occupies 2pt of vertical space. Without padding, these lines get filtered out during the height < 0.1 check.
2.  **Coordinate parity:** The rich vector extractor (pikepdf-based) expands bounding boxes by stroke width. The stream-order placeholder extractor must do the same, or the coordinates won't match during hydration. A mismatch of just a few points causes matching to fail.

**Real impact:** A 151×151 gradient logo with 1pt stroke:
* Rich vector: 151×151 (with padding)
* Placeholder without padding: 149×149
* Distance: 5.97 units > 5.0 threshold → match fails → logo appears as orphan

**A subtle trap:** `pdfminer`'s `gstate.linewidth` defaults to 0, not 1.0. The code must explicitly check for zero and apply the PDF spec default of 1.0:

```python
stroke_width = getattr(gstate, 'linewidth', 1.0)
if stroke_width == 0:
    stroke_width = 1.0  # PDF spec default

x -= stroke_width
y -= stroke_width
width += stroke_width * 2
height += stroke_width * 2
```

**Components affected:** Vector extraction, stream order extraction

---

## Vector Removal & Content Modification

### Two-Pass Vector Removal Strategy

Removing vector graphics from a PDF is significantly more complex than removing text because vectors are constructed via sequences of operators, not single commands.

**The Problem:** A vector shape is defined by a sequence like:
```
10 10 m (move to start)
20 20 l (line to)
re (rectangle)
f (fill - the "painting" operator)
```
If you blindly remove the `re` operator but leave the `f` operator, the `f` command might apply to the *previous* path in the stack or even the entire page media box, resulting in a "ghost fill" where the page turns a solid color.

**The Solution:** We implement a **Two-Pass Architecture**:
1.  **Pass 1 (Identification):** Scan the entire content stream to identify "Path Sequences." A sequence tracks all construction operators until a painting operator (`S`, `f`, `B`, etc.) is reached. We calculate the bounding box for the entire sequence. If this bbox matches a removal request, we mark the indices of every operator in that sequence for deletion.
2.  **Pass 2 (Filtering):** Re-iterate through the stream and write out only the operators that were **NOT** marked in Pass 1.

This ensures cleanly removing the entire visual element without corrupting the graphics state stack.

**Components affected:** Content modifier

### Adaptive Bounding Box Matching

When identifying which vectors to remove based on a user's selection, standard Intersection-over-Union (IoU) matching fails for thin lines or small shapes.

**The Strategy:** We use a **Multi-Tier Matching Algorithm**:
* **Tier 1 (Lines):** For elements thinner than 5px, we use Center-Point Distance with a strict 75% threshold (point must be within the central 75% of the target). IoU is useless for lines as their area is near zero.
* **Tier 2 (Small Elements):** For areas < 300px², we use Center-Point Distance with a 60% threshold.
* **Tier 3 (Standard):** For normal shapes, we use standard IoU with adaptive thresholds (0.4 - 0.7 depending on overlap).
* **Tier 4 (Fallback):** A loose proximity check for edge cases.

This allows a single API to robustly handle removing a massive background image and a tiny 1px separator line.

**Components affected:** Content modifier

---

## Form XObject Processing

Form XObjects are reusable graphics containers—think logos, headers, or decorative elements that appear on multiple pages. They're invoked with the `Do` operator and can nest arbitrarily deep.

**The problem:** PDFMiner's callback architecture (`paint_path()`, `render_string()`) doesn't fire for operators inside Form XObjects. In typical PDFs, this means 12-13% of vectors become orphans—detected by the rich vector extractor but invisible to stream-order processing.

**Why callbacks don't work:** Form XObjects are external operator streams. PDFMiner processes the main content stream but doesn't automatically recurse into XObject content unless explicitly told to.

**The solution:** Manual operator parsing. Instead of relying on callbacks, we:
1.  Detect `Do` operators that reference Form XObjects.
2.  Extract the XObject's content stream.
3.  Apply the XObject's transformation matrix.
4.  Load the XObject's local resources.
5.  Recursively parse every operator in the XObject stream.
6.  Maintain the transformation stack correctly.

This gives us complete visibility into all graphics operations, regardless of nesting depth.

**Components affected:** Vector extraction

---

## Pattern Fill Detection

PDF vectors can use Type 1 Tiling Patterns as fills. These patterns contain embedded XObject images that tile across the shape. Without pattern detection, these vectors appear with solid fills or missing fills entirely. The frontend can't reconstruct the original appearance.

**Detection criteria:**
* Path uses a fill operator (`f`, `f*`, `B`, `B*`)
* The current fill color references a Pattern resource (not a direct color)
* The Pattern dictionary has a `/PatternType` key
* For Type 1 patterns: Extract embedded images from `/Resources/XObject`

**Why it matters for rendering:** Konva (the frontend canvas library) doesn't support pattern fills natively. When we detect a pattern fill, we mark the vector for rasterization instead of path rendering. Simple solid fills can use higher-quality vector paths.

**Data structure:**
```json
{
    "type": "vector",
    "fillType": "pattern",
    "patternType": 1,
    "patternImages": [
        {
            "name": "I1",
            "width": 100,
            "height": 100,
            "data": "data:image/png;base64,..."
        }
    ]
}
```

**Components affected:** Vector extraction

---

## Sparse Vector Splitting

A 612×1591 background layer might contain only five small decorative elements. Without splitting, selecting one element targets the entire massive bounding box, making the visual editor unusable.

**The algorithm:**
1.  Calculate actual path coverage area vs. bounding box area.
2.  If coverage ratio < 30%, split into sub-paths.
3.  Group nearby sub-paths (within 10pt) back together.
4.  Create separate vector elements for each group.

**Example:**
* **Before:** 612×1591 bbox with 5 shapes (2% coverage)
* **After:** 5 separate 50×50 vectors

**Configuration:**
```python
split_sparse_vectors=True,
sparse_coverage_threshold=0.3,  # Split if <30%
sparse_merge_distance=10.0      # Merge within 10pt
```

**Components affected:** Vector processing

---

## Text Processing

### Text Matrix Advancement

Accurate word spacing requires tracking where each character starts and where it ends. The gap between words is: `current_start - previous_end`. If you only track starting positions, gap calculation fails. Variable-width fonts collapse: "hello" + "world" becomes "helloworld".

**The solution:** Calculate the ending position using the character width from the font metrics:

```python
tm_start_x = textstate.matrix[4]  # Starting X position

char_width = font.char_width(cid)
scaling = getattr(textstate, 'scaling', 100.0)
tm_end_x = tm_start_x + (char_width * font_size * scaling / 100.0)

char_data = {
    'tm_start_x': tm_start_x,  # Where character starts
    'tm_end_x': tm_end_x,      # Where character ends (CRITICAL)
}
```

Now gap calculation works correctly:
```python
gap = current_char['tm_start_x'] - previous_char['tm_end_x']
if gap > threshold:
    # Start new word
```

**Components affected:** Text extraction, text grouping

### Faux-Bold Detection

PDFs simulate bold text by rendering the same string twice at nearly identical positions, offset by ~0.5 units. Without deduplication, extracted text appears doubled. Fonts lacking true bold variants (many custom fonts) use this rendering trick universally.

**Detection:**
```python
current_position = (textstate.matrix[4], textstate.matrix[5])

if (last_text == current_text and
    abs(last_position[0] - current_position[0]) < 0.5 and
    abs(last_position[1] - current_position[1]) < 0.5):
    # Skip duplicate
    return
```

**Components affected:** Text extraction

### Cross-Run Quote Fixing

PDFs encode quotes incorrectly across text runs: `/word` (opening quote appears as slash) in one run, `word0` (closing quote appears as zero) in the next run.

When quotes span run boundaries, both runs must be tracked and fixed together:
```python
# Run 1: "He said /hello"    
# Run 2: "world0 today"

# After fixing:
# Run 1: "He said "hello"
# Run 2: "world" today"
```
The system maintains pending state between flushes. When a run ends with an unclosed `/word`, it's held until the next run arrives. If the next run starts with `word0`, both are fixed simultaneously.

**Components affected:** Text extraction

### Marker Font Filtering

PDFs contain invisible text with microscopic font sizes (<0.01pt) for metadata, OCR layers, and structural markers.

**Detection criteria:**
* **Font size < 1.0pt:** Too small for human vision.
* **Font size ≈ 0.01pt:** Common metadata marker size.

These elements shouldn't appear in extracted content. OCR layers provide searchability but aren't visually rendered. Metadata markers contain structural information for assistive technology.

**Components affected:** Text extraction

### Text Grouping Configuration

Raw character extraction produces flat lists. Grouping reconstructs document structure—lines, paragraphs, sections—for reflowable layout.

**Critical parameter:** `max_horizontal_gap=10`
This threshold determines word boundaries. Why 10?
* Typical word gap: ~0.5em ≈ 6pt at 12pt font.
* Too small (5pt): Words merge → "helloworld".
* Too large (20pt): Words fragment → "h e l l o".
* Sweet spot (10pt): Handles most PDFs correctly.

**Grouping stages:**
1.  **Horizontal grouping:** Merge characters → words (gap-based).
2.  **Line detection:** Group words on same baseline.
3.  **Paragraph detection:** Group lines with consistent vertical spacing.
4.  **Semantic analysis:** Detect lists, TOCs, headings (pattern-based).

**Components affected:** Text grouping

---

## Resource Management

### Reference Counting for Shared Resources

Images and Form XObjects can appear on multiple pages. Deleting an image from page 1 that also appears on page 10 would break page 10 if the resource is removed from the PDF.

**The algorithm:**

1.  **First pass:** Scan all pages, counting XObject references
    ```python
    image_usage_counts = {
        "Im1": 3,  # Used on pages 1, 5, 10
        "Im2": 1,  # Used only on page 3
    }
    ```
2.  **Removal:** When content is deleted, decrement reference counts.
3.  **Second pass:** Remove only resources with zero references.
    ```python
    for resource_name in page.Resources.XObject.keys():
        if image_usage_counts[resource_name] == 0:
            del page.Resources.XObject[resource_name]
    ```

This ensures shared resources survive until all references are removed.

**Components affected:** Content modification

---

## Graphics State

### Graphics State Tracking

PDF operators modify graphics state through save/restore operations (`q`/`Q`). The state affects all subsequent operations:
* **CTM:** Position, rotation, scale, skew
* **Colors:** Fill and stroke colors
* **Opacity:** Fill and stroke alpha
* **Line properties:** Width, join, cap, dash pattern

State can nest arbitrarily deep. Each `q` pushes state onto a stack; each `Q` pops it back.

**Implementation approach:**
```python
class GraphicsStateTracker:
    def __init__(self):
        self.ctm = identity_matrix()
        self.state_stack = []
        self.non_stroking_color = (0, 0, 0)
    
    def save_state(self):  # q operator
        state = {
            'ctm': self.ctm.copy(),
            'non_stroking_color': self.non_stroking_color,
            # ... all state variables
        }
        self.state_stack.append(state)
    
    def restore_state(self):  # Q operator
        if self.state_stack:
            state = self.state_stack.pop()
            self.ctm = state['ctm']
            self.non_stroking_color = state['non_stroking_color']
```

**Operators tracked:** `q`/`Q` (save/restore), `cm` (modify CTM), `g`/`G` (gray color), `rg`/`RG` (RGB), `k`/`K` (CMYK)

**Components affected:** Image extraction, vector extraction, content modification

---

## Edge Cases & Gotchas

### PDF Specification Quirks
* **pdfminer defaults:** `gstate.linewidth` defaults to 0, not 1.0. Always check and apply the PDF spec default.
* **MediaBox positioning:** The spec says "origin at the lower-left corner of the page," not "origin of the coordinate space." Pages can be positioned anywhere.
* **Image coordinate space:** Images are defined on a unit square (0,0)→(1,1), transformed by CTM to final position.
* **Form XObject nesting:** Form XObjects can contain other Form XObjects. Transformation matrices compose through the nesting chain.
* **Pattern resources:** Patterns have their own resource dictionaries separate from the page resources.

### Common Pitfalls
* **Removing quotation marks during regex:** Using `re.sub()` on text with quotes requires proper escaping or raw strings to avoid breaking quote-fixing logic.
* **Assuming sequential stream order:** PDF content streams aren't required to paint in visual order. Z-index must be explicitly tracked.
* **Coordinate system confusion:** PDF uses Y-up (origin at bottom-left), web uses Y-down (origin at top-left). Every coordinate needs conversion.
* **One-based page numbering:** PDF page numbers are 1-based, but Python list indices are 0-based. Off-by-one errors are common.
* **Font name variations:** PDF internal font names don't map directly to CSS. Requires explicit font mapping tables.