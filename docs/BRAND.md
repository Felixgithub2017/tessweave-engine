# TessWeave identity and compatibility

**TessWeave** combines the idea of a tessellated tensor tile with weaving an execution path. **Engine** identifies the independent inference runtime; **Studio** identifies the visual model-engineering control plane.

The original geometric mark shows tile-like lanes converging into a forward chevron. Midnight blue `#0B1224`, tensor cyan `#65F5DC`, and violet `#9295FF` form the palette. The mark remains static, readable at small sizes, and independent of external fonts or image services. Banner graphics are conceptual branding, not live telemetry.

- `assets/brand/mark.svg`: scalable square mark, suitable for icons.
- `assets/brand/banner.svg`: dark repository header with product descriptor.
- Studio packages the same mark as `workbench/static/tessweave.svg`.

These project-authored SVG assets use the repository's MIT license. Do not use the identity to imply upstream vendor endorsement or unmeasured speedups. A preliminary exact-name web search on 2026-10-08 found no result for TessWeave; this is **not** trademark clearance or a guarantee of name/domain availability.

## Compatibility

| Display name | Repository | Existing command/module |
| --- | --- | --- |
| TessWeave Engine | `Felixgithub2017/tessweave-engine` | `flow-inference`, `python -m flow_engine.cli` |
| TessWeave Studio | `Felixgithub2017/tessweave-studio` | `model-workbench`, `python -m workbench` |
| TessWeave Train | `Felixgithub2017/tessweave-train` | `tessweave-train`, `python -m tessweave_train.cli` |

On 2026-10-08, the GitHub repositories were renamed from `flow-inference` and `model-workbench` to `tessweave-engine` and `tessweave-studio`. Local checkout folders, imports, state, metrics and existing CLI commands remain unchanged. Historical logs, filenames and the system-report draft may retain “Flow Inference” or “Model Workbench”; these refer to the same respective projects. The projects do not require one another to be installed. Train integration remains a roadmap item.
