# Before and after

Same states, same script (`e2e/redesign.spec.ts`, `bun run screenshots:redesign`; `REDESIGN_DIR` picks the
folder). Every state has four before | after pairs: desktop light, desktop dark, phone light, phone dark. Plan:
[plan.md](plan.md); iteration log: [notes.md](notes.md).

## Default load

Inputs move to the right third; the setup becomes a serif headline; figures, roofline, verdict and the top of the table now fit above the fold under a 48px deep-blue masthead.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Default load, desktop light, before](before/01-default-desktop-light.png) | ![Default load, desktop light, after](after/01-default-desktop-light.png) |
| Desktop dark | ![Default load, desktop dark, before](before/01-default-desktop-dark.png) | ![Default load, desktop dark, after](after/01-default-desktop-dark.png) |
| Phone light | ![Default load, phone light, before](before/01-default-phone-light.png) | ![Default load, phone light, after](after/01-default-phone-light.png) |
| Phone dark | ![Default load, phone dark, before](before/01-default-phone-dark.png) | ![Default load, phone dark, after](after/01-default-phone-dark.png) |

## A fitting config

The fit verdict is an ink line over a quiet track, so the danger red is kept for real overflow; the headline names the link ("over PCIe 5.0 x16 with host staging").

| Variant | Before | After |
|---|---|---|
| Desktop light | ![A fitting config, desktop light, before](before/02-fitting-desktop-light.png) | ![A fitting config, desktop light, after](after/02-fitting-desktop-light.png) |
| Desktop dark | ![A fitting config, desktop dark, before](before/02-fitting-desktop-dark.png) | ![A fitting config, desktop dark, after](after/02-fitting-desktop-dark.png) |
| Phone light | ![A fitting config, phone light, before](before/02-fitting-phone-light.png) | ![A fitting config, phone light, after](after/02-fitting-phone-light.png) |
| Phone dark | ![A fitting config, phone dark, before](before/02-fitting-phone-dark.png) | ![A fitting config, phone dark, after](after/02-fitting-phone-dark.png) |

## DGX Spark ×4 ring

Machines segments, Network link and Cabling share one 36px control size and a two-up row; the headline keeps "4 × NVIDIA" together.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![DGX Spark ×4 ring, desktop light, before](before/03-multi-node-desktop-light.png) | ![DGX Spark ×4 ring, desktop light, after](after/03-multi-node-desktop-light.png) |
| Desktop dark | ![DGX Spark ×4 ring, desktop dark, before](before/03-multi-node-desktop-dark.png) | ![DGX Spark ×4 ring, desktop dark, after](after/03-multi-node-desktop-dark.png) |
| Phone light | ![DGX Spark ×4 ring, phone light, before](before/03-multi-node-phone-light.png) | ![DGX Spark ×4 ring, phone light, after](after/03-multi-node-phone-light.png) |
| Phone dark | ![DGX Spark ×4 ring, phone dark, before](before/03-multi-node-phone-dark.png) | ![DGX Spark ×4 ring, phone dark, after](after/03-multi-node-phone-dark.png) |

## Open device picker

The popup has a white surface with a blue-tinted shadow and a 36px search box; the active option has a tint plus a 2px blue edge instead of a grey wash.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Open device picker, desktop light, before](before/04-device-picker-desktop-light.png) | ![Open device picker, desktop light, after](after/04-device-picker-desktop-light.png) |
| Desktop dark | ![Open device picker, desktop dark, before](before/04-device-picker-desktop-dark.png) | ![Open device picker, desktop dark, after](after/04-device-picker-desktop-dark.png) |
| Phone light | ![Open device picker, phone light, before](before/04-device-picker-phone-light.png) | ![Open device picker, phone light, after](after/04-device-picker-phone-light.png) |
| Phone dark | ![Open device picker, phone dark, before](before/04-device-picker-phone-dark.png) | ![Open device picker, phone dark, after](after/04-device-picker-phone-dark.png) |

## Open "?" help

The 15px "?" sits 6px after its label, centred on the cap height, with a 24px hit area that does not move the field; the tooltip is a light card next to its icon.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Open "?" help, desktop light, before](before/05-input-help-desktop-light.png) | ![Open "?" help, desktop light, after](after/05-input-help-desktop-light.png) |
| Desktop dark | ![Open "?" help, desktop dark, before](before/05-input-help-desktop-dark.png) | ![Open "?" help, desktop dark, after](after/05-input-help-desktop-dark.png) |
| Phone light | ![Open "?" help, phone light, before](before/05-input-help-phone-light.png) | ![Open "?" help, phone light, after](after/05-input-help-phone-light.png) |
| Phone dark | ![Open "?" help, phone dark, before](before/05-input-help-phone-dark.png) | ![Open "?" help, phone dark, after](after/05-input-help-phone-dark.png) |

## Spec-dec knobs on

The speedup sits under the decode figure with its "?" on the same line; spec-dec fields use the standard label → control rhythm.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Spec-dec knobs on, desktop light, before](before/06-spec-dec-desktop-light.png) | ![Spec-dec knobs on, desktop light, after](after/06-spec-dec-desktop-light.png) |
| Desktop dark | ![Spec-dec knobs on, desktop dark, before](before/06-spec-dec-desktop-dark.png) | ![Spec-dec knobs on, desktop dark, after](after/06-spec-dec-desktop-dark.png) |
| Phone light | ![Spec-dec knobs on, phone light, before](before/06-spec-dec-phone-light.png) | ![Spec-dec knobs on, phone light, after](after/06-spec-dec-phone-light.png) |
| Phone dark | ![Spec-dec knobs on, phone dark, before](before/06-spec-dec-phone-dark.png) | ![Spec-dec knobs on, phone dark, after](after/06-spec-dec-phone-dark.png) |

## Offload knobs on

Offload is a sub-heading with space instead of a rule; offload chips are mist tags; the host RAM tier bar matches the device bar.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Offload knobs on, desktop light, before](before/07-offload-desktop-light.png) | ![Offload knobs on, desktop light, after](after/07-offload-desktop-light.png) |
| Desktop dark | ![Offload knobs on, desktop dark, before](before/07-offload-desktop-dark.png) | ![Offload knobs on, desktop dark, after](after/07-offload-desktop-dark.png) |
| Phone light | ![Offload knobs on, phone light, before](before/07-offload-phone-light.png) | ![Offload knobs on, phone light, after](after/07-offload-phone-light.png) |
| Phone dark | ![Offload knobs on, phone dark, before](before/07-offload-phone-dark.png) | ![Offload knobs on, phone dark, after](after/07-offload-phone-dark.png) |

## Chart tooltip

The roofline has no boxy axes, only light gridlines; the tooltip is a white card with a 1px border and units on every value.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Chart tooltip, desktop light, before](before/08-chart-tooltip-desktop-light.png) | ![Chart tooltip, desktop light, after](after/08-chart-tooltip-desktop-light.png) |
| Desktop dark | ![Chart tooltip, desktop dark, before](before/08-chart-tooltip-desktop-dark.png) | ![Chart tooltip, desktop dark, after](after/08-chart-tooltip-desktop-dark.png) |
| Phone light | ![Chart tooltip, phone light, before](before/08-chart-tooltip-phone-light.png) | ![Chart tooltip, phone light, after](after/08-chart-tooltip-phone-light.png) |
| Phone dark | ![Chart tooltip, phone dark, before](before/08-chart-tooltip-phone-dark.png) | ![Chart tooltip, phone dark, after](after/08-chart-tooltip-phone-dark.png) |

## Memory overflow

The overflow verdict is 18px semibold crimson with the glyph on its line; the overflowing part of the bar is solid crimson.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![Memory overflow, desktop light, before](before/09-memory-overflow-desktop-light.png) | ![Memory overflow, desktop light, after](after/09-memory-overflow-desktop-light.png) |
| Desktop dark | ![Memory overflow, desktop dark, before](before/09-memory-overflow-desktop-dark.png) | ![Memory overflow, desktop dark, after](after/09-memory-overflow-desktop-dark.png) |
| Phone light | ![Memory overflow, phone light, before](before/09-memory-overflow-phone-light.png) | ![Memory overflow, phone light, after](after/09-memory-overflow-phone-light.png) |
| Phone dark | ![Memory overflow, phone dark, before](before/09-memory-overflow-phone-dark.png) | ![Memory overflow, phone dark, after](after/09-memory-overflow-phone-dark.png) |

## High utilisation

Burnt-orange verdict and "?" on one line; the figures carry a 16px warning glyph after their unit, not between number and unit.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![High utilisation, desktop light, before](before/10-memory-high-desktop-light.png) | ![High utilisation, desktop light, after](after/10-memory-high-desktop-light.png) |
| Desktop dark | ![High utilisation, desktop dark, before](before/10-memory-high-desktop-dark.png) | ![High utilisation, desktop dark, after](after/10-memory-high-desktop-dark.png) |
| Phone light | ![High utilisation, phone light, before](before/10-memory-high-phone-light.png) | ![High utilisation, phone light, after](after/10-memory-high-phone-light.png) |
| Phone dark | ![High utilisation, phone dark, before](before/10-memory-high-phone-dark.png) | ![High utilisation, phone dark, after](after/10-memory-high-phone-dark.png) |

## About page

Same masthead; the methodology reads in Newsreader at a 64ch measure; code chips no longer use spaced tabular colons.

| Variant | Before | After |
|---|---|---|
| Desktop light | ![About page, desktop light, before](before/11-about-desktop-light.png) | ![About page, desktop light, after](after/11-about-desktop-light.png) |
| Desktop dark | ![About page, desktop dark, before](before/11-about-desktop-dark.png) | ![About page, desktop dark, after](after/11-about-desktop-dark.png) |
| Phone light | ![About page, phone light, before](before/11-about-phone-light.png) | ![About page, phone light, after](after/11-about-phone-light.png) |
| Phone dark | ![About page, phone dark, before](before/11-about-phone-dark.png) | ![About page, phone dark, after](after/11-about-phone-dark.png) |

## Across every state

- **Layout:** one container split by a 1px draggable rule, output left 2/3, inputs right 1/3, each pane in its own
  themed ScrollArea; the mini result bar pins prefill and decode (with their context) once the hero scrolls away.
  Phone: headline → figures → memory → inputs (collapsed sections with their value) → roofline and analysis,
  with the mini bar pinned at the top while editing.
- **Type:** Newsreader for the headline and section heads, Schibsted Grotesk for everything else, digits tabular
  with natural-width separators.
- **Colour:** deep blue (`#0F3B82` light, `#8FB4F0` on dark) only for the masthead, links, focus, selection and the
  splitter; greys tuned to the blue; crimson and burnt orange for fit risk; the three bound colours unchanged in
  role (teal and ochre darkened to pass AA).
- **Sizes:** controls 36px (44px on phones), "?" 15px with a 24px hit area, label → control 6px, field → field
  16px, section → section 32px (inputs) and 48px (output).
