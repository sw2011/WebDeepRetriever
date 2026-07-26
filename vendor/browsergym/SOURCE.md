# BrowserGym vendored source

- Upstream project: BrowserGym
- Upstream repository: https://github.com/ServiceNow/BrowserGym
- Upstream commit: `9e779f087de9a65668b6974d11f9ce9816026e96`
- Upstream package version at that commit: `0.14.3`
- License: Apache License 2.0
- Retrieved: 2026-07-26
- Local source root: `vendor/browsergym/src`

The files below are copied byte-for-byte from
`browsergym/core/src/browsergym` at the pinned commit. They intentionally form
a small source subset rather than a vendored copy of the complete BrowserGym
package.

| Local file | SHA-256 | Inclusion reason |
| --- | --- | --- |
| `src/browsergym/core/observation.py` | `c3e273ddc222cfd3eaaa2346acfa991df4d77285befee40bc17ee142203a557c` | DOMSnapshot extraction, merged iframe AXTree extraction, stable BrowserGym IDs, and screenshot extraction. |
| `src/browsergym/core/action/utils.py` | `8ca45f9b620bd576f9e6c61fe0912282aec3b9d295ba1ad16c7d837372f5974b` | Playwright locator lookup by BrowserGym ID, including nested iframe traversal. |
| `src/browsergym/core/constants.py` | `aa4a9eb93601dc8c8fd4a4d3ca9ed752503314aaa21c50685ab71657828c5f43` | Internal dependency that defines the temporary DOM attribute names shared by observation and formatting code. |
| `src/browsergym/core/javascript/frame_mark_elements.js` | `6ecfbba89d756c37874bdf692e4bf24126459ab79f9120c1f053305b7ba2978a` | Runtime package resource loaded by `observation.py`; assigns stable IDs and traverses open Shadow DOM. |
| `src/browsergym/core/javascript/frame_unmark_elements.js` | `ee1324efd8df5c12693fba0c4814ecb829ef92860d972e5377bdd5c905b28526` | Runtime package resource loaded by `observation.py`; removes temporary page annotations after extraction. |
| `src/browsergym/utils/obs.py` | `eb08e2c263cf78c495ed942129bf6eab7da744e75b1213ed15bf6a13d92b79d7` | Converts DOMSnapshot and AXTree data to bounded textual observations and supports set-of-marks rendering. |

No upstream source file in this subset has been modified. No BrowserGym action
parser or Python-code execution module is included.

## Dependency closure

The only imports between the selected BrowserGym files are from
`browsergym.utils.obs` and `browsergym.core.observation` to
`browsergym.core.constants`. `observation.py` also loads both JavaScript files
through `pkgutil.get_data`. Those three dependencies are included above.

The selected Python modules additionally import these third-party runtime
packages, which are not copied into this directory: Playwright, NumPy, Pillow,
Beautiful Soup, and lxml (used by Beautiful Soup in `prune_html`). Python
standard-library imports require no vendored files.

The upstream `browsergym.core.__init__`, environment, registration, task,
spaces, and high-level action modules are deliberately excluded: the selected
modules do not import them, and including them would pull in the full Gymnasium
environment and task-registration stack.

The upstream license text is preserved at `vendor/browsergym/LICENSE`. Project-
level attribution is recorded in `THIRD_PARTY_NOTICES.md`.
