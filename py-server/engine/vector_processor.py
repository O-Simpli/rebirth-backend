"""
PDF Vector Processor

High-level processor for extracting vector graphics from PDFs.
Coordinates vector extraction using the VectorExtractionDevice.
"""

import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from engine.base_processor import BaseProcessor
from engine.config import VectorProcessorOptions
from processors.vector_extraction_device import VectorExtractionDevice, stable_id, extract_path_data
from models.pdf_types import PdfVectorElement

if TYPE_CHECKING:
    from engine.pdf_engine import PDFEngine

logger = logging.getLogger(__name__)


class VectorProcessor(BaseProcessor):
    """
    Processor for extracting vector graphics from PDFs.
    
    This processor wraps the VectorExtractionDevice and provides a high-level API
    for extracting vectors with page-level coordination.
    
    Integrated with PDFEngine for resource management and lifecycle control.
    """

    def __init__(self, engine: 'PDFEngine', options: Optional[VectorProcessorOptions] = None):
        """
        Initialize vector processor.
        
        Args:
            engine: PDFEngine instance for resource management
            options: Vector processor configuration options
        """
        super().__init__(engine)
        self.options = options or VectorProcessorOptions()

    def initialize(self) -> None:
        """Initialize vector processor resources."""
        super().initialize()
        logger.debug("VectorProcessor initialized")

    def cleanup(self) -> None:
        """Clean up vector processor resources."""
        super().cleanup()
        logger.debug("VectorProcessor cleaned up")

    def extract_vectors(
        self,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> List[List[PdfVectorElement]]:
        """
        Extract vector graphics from PDF pages.

        Args:
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed), None for all pages

        Returns:
            List of vector elements per page
            
        Raises:
            RuntimeError: If processor not initialized
        """
        if not self.validate_state():
            raise RuntimeError("VectorProcessor not initialized. Use PDFEngine context manager.")
        
        try:
            # Use engine's pikepdf document instead of opening a new one
            pdf = self.engine.pikepdf_document
            pdf_path = self.engine.file_path
            
            total_pages = len(pdf.pages)
            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page) if end_page else total_pages

            logger.info(f"Extracting vectors from pages {start_page} to {end_idx} of {total_pages}")

            all_vectors: List[List[PdfVectorElement]] = []

            for page_idx in range(start_idx, end_idx):
                page = pdf.pages[page_idx]
                page_num = page_idx + 1

                try:
                    vectors = self._extract_page_vectors(pdf_path, page, page_num)
                    all_vectors.append(vectors)
                    logger.debug(f"Page {page_num}: extracted {len(vectors)} vectors")
                except Exception as e:
                    logger.error(f"Error extracting vectors from page {page_num}: {e}")
                    all_vectors.append([])

            if self.options.enable_vector_grouping:
                all_vectors = self._group_vectors(all_vectors)

            return all_vectors

        except Exception as e:
            logger.error(f"Failed to extract vectors: {e}")
            raise

    def _extract_page_vectors(
        self,
        pdf_path: str,
        page,
        page_num: int,
    ) -> List[PdfVectorElement]:
        """
        Extract vectors from a single page.

        Args:
            pdf_path: Path to PDF file
            page: pikepdf Page object
            page_num: Page number (1-indexed)

        Returns:
            List of vector elements
        """
        # Get page height for coordinate transformation
        mediabox = page.MediaBox
        page_height = float(mediabox[3] - mediabox[1])
        mediabox_bottom = float(mediabox[1])

        # Get config options
        include_pattern_image_data = self.options.include_pattern_image_data
        generate_svg = self.options.generate_svg
        generate_path_data = self.options.generate_path_data
        split_sparse_vectors = self.options.split_sparse_vectors
        sparse_coverage_threshold = self.options.sparse_vector_coverage_threshold
        sparse_merge_distance = self.options.sparse_vector_merge_distance
        
        # Get overlap merging options
        enable_overlap_grouping = self.options.enable_overlap_grouping
        overlap_check_window = self.options.overlap_check_window
        overlap_method = self.options.overlap_method
        overlap_threshold = self.options.overlap_threshold
        proximity_threshold = self.options.proximity_threshold
        enable_post_processing_merge = self.options.enable_post_processing_merge

        # Create extraction device with performance flags and splitting/grouping options
        device = VectorExtractionDevice(
            page_height=page_height,
            mediabox_bottom=mediabox_bottom,
            include_pattern_image_data=include_pattern_image_data,
            generate_svg=generate_svg,
            generate_path_data=generate_path_data,
            split_sparse_vectors=split_sparse_vectors,
            sparse_coverage_threshold=sparse_coverage_threshold,
            sparse_merge_distance=sparse_merge_distance,
            enable_overlap_grouping=enable_overlap_grouping,
            overlap_check_window=overlap_check_window,
            overlap_method=overlap_method,
            overlap_threshold=overlap_threshold,
            proximity_threshold=proximity_threshold,
            enable_post_processing_merge=enable_post_processing_merge
        )

        # Extract vectors (with stream-based splitting and grouping if enabled)
        device.extract(page)

        # Convert to PdfVectorElement
        result: List[PdfVectorElement] = []
        for i, vec_dict in enumerate(device.vectors):
            try:
                # Generate stable ID
                if vec_dict.get("_splitFrom") == "stream":
                    # Stream-split sub-vector - include bounds for uniqueness
                    split_idx = vec_dict.get("_splitIndex", 0)
                    vec_id = stable_id(
                        pdf_path, 
                        page_num - 1, 
                        "split",
                        split_idx,
                        round(vec_dict["x"], 2),
                        round(vec_dict["y"], 2),
                        round(vec_dict["width"], 2),
                        round(vec_dict["height"], 2)
                    )
                else:
                    # Regular vector - use index
                    vec_id = stable_id(pdf_path, page_num - 1, i)

                # Extract path data from SVG for client-side rendering (only if both SVG and path data requested)
                path_data = None
                if generate_path_data and generate_svg and vec_dict.get("svgContent"):
                    path_data = extract_path_data(vec_dict["svgContent"])

                # Create PdfVectorElement
                vector_elem = PdfVectorElement(
                    id=vec_id,
                    type="vector",
                    x=vec_dict["x"],
                    y=vec_dict["y"],
                    width=vec_dict["width"],
                    height=vec_dict["height"],
                    svgContent=vec_dict["svgContent"] if generate_svg else "",
                    pathData=path_data,
                    opacity=1.0,  # Base opacity (fill/stroke opacity in SVG)
                    stroke=vec_dict.get("stroke"),
                    fill=vec_dict.get("fill"),
                    strokeWidth=vec_dict.get("strokeWidth"),
                    fillType=vec_dict.get("fillType"),
                    patternType=vec_dict.get("patternType"),
                    patternImages=vec_dict.get("patternImages"),
                    clippingRegionId=vec_dict.get("clippingRegionId"),
                    clippingPath=vec_dict.get("clippingPath"),
                    clippingRule=vec_dict.get("clippingRule"),                    
                    stream_index=vec_dict.get("stream_index"),
                )

                result.append(vector_elem)

            except Exception as e:
                logger.warning(f"Failed to create vector element {i} on page {page_num}: {e}")
                continue

        return result

    def _group_vectors(self, all_vectors: List[List[PdfVectorElement]]) -> List[List[PdfVectorElement]]:
        """
        Group vectors intelligently using a two-phase approach:
        
        Phase 1: Group vectors by clipping region ID (explicit PDF structure)
        - Vectors with the same clippingRegionId are merged into one
        
        Phase 2: Merge spatially overlapping groups (visual relationships)
        - Groups that spatially overlap are merged together
        - This catches related graphics split across different clipping regions
        
        Args:
            all_vectors: List of vector lists (one per page)
        
        Returns:
            List of vector lists with intelligently grouped vectors
        """
        from collections import defaultdict
        
        result = []
        
        for page_vectors in all_vectors:
            if not page_vectors:
                result.append([])
                continue
            
            logger.debug(f"Grouping page with {len(page_vectors)} vectors")
            
            # Phase 1: Group by clipping region
            clipping_groups: dict[int | None, list[PdfVectorElement]] = defaultdict(list)
            for vector in page_vectors:
                clipping_groups[vector.clippingRegionId].append(vector)
            
            logger.debug(f"Found {len(clipping_groups)} clipping groups")
            
            # Phase 2: Collect clipping groups without merging yet
            # Keep vectors as lists to preserve stream_index
            grouped_vectors = []
            
            for clip_id, vectors_in_group in clipping_groups.items():
                if clip_id is None:
                    # No clipping region - keep each vector separate
                    for vector in vectors_in_group:
                        grouped_vectors.append([vector])
                else:
                    # Group vectors by clipping region, but don't merge yet
                    grouped_vectors.append(vectors_in_group)
                    logger.debug(f"Grouped {len(vectors_in_group)} vectors from clipping region {clip_id}")
            
            # Phase 3: Detect spatial overlaps and build super-groups (no merging yet)
            super_groups = self._detect_overlapping_groups(grouped_vectors)
            
            # Phase 4: Flatten, sort by stream_index, and merge once
            final_vectors = []
            for group in super_groups:
                # Flatten all vectors from this super-group
                flattened_vectors = []
                for subgroup in group:
                    flattened_vectors.extend(subgroup)
                
                # Sort by stream_index to preserve z-order
                sorted_vectors = sorted(
                    flattened_vectors, 
                    key=lambda v: v.stream_index if v.stream_index is not None else float('inf')
                )
                
                if len(sorted_vectors) == 1:
                    # Single vector, no merge needed
                    final_vectors.append(sorted_vectors[0])
                else:
                    # Merge with earliest stream_index preserved
                    merged = self._merge_and_preserve_order(sorted_vectors)
                    final_vectors.append(merged)
                    logger.debug(f"Merged {len(sorted_vectors)} vectors with stream_index={merged.stream_index}")
            
            result.append(final_vectors)
            logger.debug(f"After spatial grouping: {len(final_vectors)} vectors")
        
        return result
    
    def _translate_svg_path(self, path_data: str, translate_x: float, translate_y: float) -> str:
        """Translate an SVG path by the given offsets."""
        import re
        
        # Parse and translate coordinates in the path data
        # SVG path commands: M/m, L/l, H/h, V/v, C/c, S/s, Q/q, T/t, A/a, Z/z
        def translate_coords(match):
            cmd = match.group(1)
            coords_str = match.group(2)
            
            # Split coordinates
            coords = [float(x) for x in re.findall(r'-?\d+\.?\d*', coords_str)]
            
            # Commands that need translation (absolute coordinates)
            if cmd in 'MLHVCSQTA':
                if cmd == 'H':
                    # Horizontal line: translate X
                    coords = [c + translate_x for c in coords]
                elif cmd == 'V':
                    # Vertical line: translate Y
                    coords = [c + translate_y for c in coords]
                elif cmd == 'A':
                    # Arc: translate only final X,Y (indices 5,6 in groups of 7)
                    for i in range(5, len(coords), 7):
                        coords[i] += translate_x
                        if i + 1 < len(coords):
                            coords[i + 1] += translate_y
                else:
                    # All other commands: translate X,Y pairs
                    for i in range(0, len(coords), 2):
                        coords[i] += translate_x
                        if i + 1 < len(coords):
                            coords[i + 1] += translate_y
            # Relative commands (lowercase) don't need translation
            
            return cmd + ' ' + ', '.join(f'{c:.3f}' for c in coords)
        
        # Match path commands followed by coordinates
        translated = re.sub(r'([MLHVCSQTAmlhvcsqta])\s*([\d\s,.-]+)', translate_coords, path_data)
        return translated
    
    def _merge_clipping_group(self, vectors: list[PdfVectorElement], clip_id: Optional[int], clipping_data: Optional[dict] = None) -> PdfVectorElement:
        """
        Merge multiple vectors from the same clipping region into a single grouped vector.
        Preserves nested SVG structure and prevents ID collisions by namespacing all defs.
        """
        import re
        
        if len(vectors) == 1:
            return vectors[0]
        
        # 1. Compute union bounding box
        min_x = min(v.x for v in vectors)
        min_y = min(v.y for v in vectors)
        max_x = max(v.x + v.width for v in vectors)
        max_y = max(v.y + v.height for v in vectors)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 2. Prepare SVG Content
        combined_content = []
        all_defs = []
        vector_clipPaths = {} 
        vector_index = 0
        
        for v in vectors:
            if not v.svgContent:
                continue
            
            try:
                # Extract inner content from existing SVGs
                svg_content = v.svgContent.strip()
                inner_content_match = re.search(r'<svg[^>]*>(.*)</svg>\s*</svg>', svg_content, re.DOTALL)
                if not inner_content_match:
                    inner_content_match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
                
                if not inner_content_match: continue
                inner_content = inner_content_match.group(1)
                
                # Namespacing Prefix for this vector instance
                prefix = f"v{vector_index}_"
                
                # Extract defs
                defs_match = re.search(r'<defs>(.*?)</defs>', inner_content, re.DOTALL)
                if defs_match:
                    defs_content = defs_match.group(1)
                    
                    # CRITICAL FIX: Find ALL IDs (gradients, clipPaths, patterns, filters) and rename them
                    # This prevents "clip_1" from Vector A colliding with "clip_1" from Vector B
                    # We look for id="..." attributes inside specific tags we expect
                    id_pattern = r'<(?:linearGradient|radialGradient|clipPath|pattern|filter)[^>]*\sid="([^"]+)"'
                    ids_to_rename = re.findall(id_pattern, defs_content)
                    
                    for old_id in ids_to_rename:
                        new_id = f"{prefix}{old_id}"
                        
                        # Replace ID definition in defs
                        # Use strictly bounded replace to avoid partial matches (e.g. id="a" vs id="aa")
                        defs_content = re.sub(f'id="{re.escape(old_id)}"', f'id="{new_id}"', defs_content)
                        
                        # Replace url(#ID) references in BOTH defs (nested) and inner_content
                        # We handle url(#id) and potentially href="#id"
                        ref_pattern = f'url\(#{re.escape(old_id)}\)'
                        new_ref = f'url(#{new_id})'
                        defs_content = re.sub(ref_pattern, new_ref, defs_content)
                        inner_content = re.sub(ref_pattern, new_ref, inner_content)
                        
                        # Also handle href="#id" (common in <use> or <pattern>)
                        href_pattern = f'href="#{re.escape(old_id)}"'
                        new_href = f'href="#{new_id}"'
                        defs_content = re.sub(href_pattern, new_href, defs_content)
                        inner_content = re.sub(href_pattern, new_href, inner_content)

                    all_defs.append(defs_content)
                
                # Setup Top-Level Clipping for this vector (The "Merge Group" Clip)
                unique_clip_id = None
                clip_attr = ""
                if v.clippingRegionId and v.clippingPath:
                    unique_clip_id = f"group_clip_{v.clippingRegionId}_{vector_index}"
                    vector_clipPaths[unique_clip_id] = {
                        'path': v.clippingPath,
                        'rule': v.clippingRule or 'nonzero',
                        'tx': -v.x, # Shift clip to vector's local space
                        'ty': -v.y
                    }
                    clip_attr = f'clip-path="url(#{unique_clip_id})"'

                # Remove defs from body (moved to global)
                inner_content_no_defs = re.sub(r'<defs>.*?</defs>', '', inner_content, flags=re.DOTALL)
                
                # Wrap content in a group positioned correctly
                translate_x = v.x - min_x
                translate_y = v.y - min_y
                
                wrapper = (
                    f'<g transform="translate({translate_x}, {translate_y})" {clip_attr}>'
                    f'{inner_content_no_defs}'
                    f'</g>'
                )
                
                combined_content.append(wrapper)
                vector_index += 1
                    
            except Exception as e:
                logger.warning(f"Failed to merge vector {v.id}: {e}")
                continue
        
        # 4. Build Final SVG
        if combined_content:
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            
            if all_defs or vector_clipPaths:
                svg_content += '<defs>'
                if all_defs:
                    svg_content += ''.join(all_defs)
                
                # Generate clip definitions with TRANSFORMS
                for unique_id, data in vector_clipPaths.items():
                    rule = "evenodd" if data['rule'] == "evenodd" else "nonzero"
                    transform = f'transform="translate({data["tx"]}, {data["ty"]})"'
                    
                    svg_content += (
                        f'<clipPath id="{unique_id}">'
                        f'<path d="{data["path"]}" clip-rule="{rule}" {transform}/>'
                        f'</clipPath>'
                    )
                
                svg_content += '</defs>'
            
            svg_content += ''.join(combined_content)
            svg_content += '</svg>'
        else:
            svg_content = vectors[0].svgContent if vectors else ""
        
        first = vectors[0]
        
        # Metadata fallback
        final_clip_path = None
        final_clip_rule = None
        if clipping_data:
            paths = [d['path'] for d in clipping_data.values() if d.get('path')]
            if paths:
                final_clip_path = paths[0]
                final_clip_rule = "evenodd"
        elif clip_id:
            for v in vectors:
                if v.clippingPath and v.clippingRegionId == clip_id:
                    final_clip_path = v.clippingPath
                    final_clip_rule = v.clippingRule
                    break

        return PdfVectorElement(
            id=f"{first.id}_clip{clip_id}",
            type="vector",
            x=min_x,
            y=min_y,
            width=width,
            height=height,
            svgContent=svg_content,
            pathData=None,
            opacity=first.opacity,
            stroke=first.stroke,
            fill=first.fill,
            strokeWidth=first.strokeWidth,
            fillType=first.fillType,
            patternType=first.patternType,
            patternImages=first.patternImages,
            clippingRegionId=clip_id,
            clippingPath=final_clip_path,
            clippingRule=final_clip_rule,
        )
    
    def _apply_overlap_grouping(self, vectors: List[PdfVectorElement]) -> List[PdfVectorElement]:
        """
        Apply overlap-based grouping to vectors without clipping regions.
        
        Args:
            vectors: List of vectors to group
        
        Returns:
            List of vectors with overlapping ones merged
        """
        # TODO: Implement overlap grouping if needed
        # For now, just return vectors as-is
        return vectors

    def _detect_overlapping_groups(self, vector_groups: list[list[PdfVectorElement]]) -> list[list[list[PdfVectorElement]]]:
        """
        Detect which vector groups spatially overlap using union-find.
        Returns super-groups (groups of groups) without merging.
        
        Args:
            vector_groups: List of vector groups (each group is a list of vectors)
        
        Returns:
            List of super-groups, where each super-group is a list of vector groups that overlap
        """
        if len(vector_groups) <= 1:
            return [vector_groups]
        
        n = len(vector_groups)
        
        # Compute representative bounding box for each group
        def get_group_bounds(group: list[PdfVectorElement]):
            if not group:
                return None
            min_x = min(v.x for v in group)
            min_y = min(v.y for v in group)
            max_x = max(v.x + v.width for v in group)
            max_y = max(v.y + v.height for v in group)
            return (min_x, min_y, max_x - min_x, max_y - min_y)
        
        group_bounds = [get_group_bounds(g) for g in vector_groups]
        
        # Union-Find parent array
        parent = list(range(n))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Check all pairs for overlap using group bounding boxes
        for i in range(n):
            for j in range(i + 1, n):
                if group_bounds[i] and group_bounds[j]:
                    # Create temporary bounding box objects for overlap check
                    class BoundingBox:
                        def __init__(self, x, y, width, height):
                            self.x = x
                            self.y = y
                            self.width = width
                            self.height = height
                    
                    box1 = BoundingBox(*group_bounds[i])
                    box2 = BoundingBox(*group_bounds[j])
                    
                    if self._boxes_overlap(box1, box2):
                        union(i, j)
        
        # Group vector groups by their root parent
        from collections import defaultdict
        super_groups: dict[int, list[list[PdfVectorElement]]] = defaultdict(list)
        for i, vector_group in enumerate(vector_groups):
            root = find(i)
            super_groups[root].append(vector_group)
        
        return list(super_groups.values())
    
    def _merge_and_preserve_order(self, vectors: list[PdfVectorElement]) -> PdfVectorElement:
        """
        Merge multiple vectors while preserving the earliest stream_index for z-order.
        
        Args:
            vectors: Sorted list of vectors to merge (should be pre-sorted by stream_index)
        
        Returns:
            Merged vector with earliest stream_index preserved
        """
        if len(vectors) == 1:
            return vectors[0]
        
        # Collect all unique clipping IDs and their paths
        clipping_data = {}
        for v in vectors:
            if v.clippingRegionId and v.clippingPath:
                clipping_data[v.clippingRegionId] = {
                    'path': v.clippingPath,
                    'rule': v.clippingRule or 'nonzero'
                }
        
        # Use a combined clip ID (hash of all clip IDs involved)
        if clipping_data:
            clip_ids_str = '_'.join(sorted(str(k) for k in clipping_data.keys()))
            import hashlib
            unified_clip_id = int(hashlib.sha256(clip_ids_str.encode()).hexdigest()[:8], 16)
        else:
            # If no clipping, use first vector's clipping ID
            unified_clip_id = vectors[0].clippingRegionId
        
        # Merge using existing method
        merged = self._merge_clipping_group(
            vectors, 
            unified_clip_id, 
            clipping_data if clipping_data else None
        )
        
        # CRITICAL: Preserve the earliest stream_index from sorted vectors
        earliest_stream_index = None
        for v in vectors:
            if v.stream_index is not None:
                if earliest_stream_index is None or v.stream_index < earliest_stream_index:
                    earliest_stream_index = v.stream_index
        
        # Update merged vector with preserved stream_index
        merged.stream_index = earliest_stream_index
        
        return merged
    
    def _boxes_overlap(self, v1, v2, threshold: float = 0.0) -> bool:
        """
        Check if two bounding boxes overlap significantly.
        Uses a conservative approach - boxes must have substantial overlap, not just touch.
        
        Args:
            v1, v2: Objects with x, y, width, height attributes
            threshold: Additional margin for "close enough" (in PDF units)
        
        Returns:
            True if boxes have significant overlap (>50% intersection area)
        """
        # Don't use margin - require actual overlap
        x1_min, x1_max = v1.x, v1.x + v1.width
        y1_min, y1_max = v1.y, v1.y + v1.height
        
        x2_min, x2_max = v2.x, v2.x + v2.width
        y2_min, y2_max = v2.y, v2.y + v2.height
        
        # Calculate intersection area
        x_overlap_start = max(x1_min, x2_min)
        x_overlap_end = min(x1_max, x2_max)
        y_overlap_start = max(y1_min, y2_min)
        y_overlap_end = min(y1_max, y2_max)
        
        # No overlap if intersection is negative
        if x_overlap_end <= x_overlap_start or y_overlap_end <= y_overlap_start:
            return False
        
        # Calculate intersection area
        intersection_area = (x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)
        
        # Calculate areas
        area1 = v1.width * v1.height
        area2 = v2.width * v2.height
        smaller_area = min(area1, area2)
        
        # Require at least 50% of the smaller box to be overlapped
        # This prevents merging things that just barely touch
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
        
        return overlap_ratio > 0.5