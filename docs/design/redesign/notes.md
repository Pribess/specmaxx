# Redesign iteration notes

What each render showed and what changed because of it. Renders: headless Chromium through
`e2e/redesign.spec.ts` (all 11 states × desktop/phone × light/dark), plus 3× device-pixel-ratio element
crops of inline details (label + "?", heading + "?", verdict + "?", chips + "?", picker triggers, the hero
figures, segmented controls) and two extra viewports (1024 × 768, 1920 × 1080) with the splitter hovered.

## Pass 1: tokens, fonts, split layout

- **Seen:** Schibsted Grotesk's `tnum` feature also makes the comma, full stop and colon digit-wide, so every
  value set tabular read "3 , 953", "Qwen3 . 8-27B", "26 . 9B total" and "qwen : qwen3_5".
  **Changed:** a `Num` component sets digits tabular and separators proportional (`.num-sep`). It is used on the
  hero figures, the mini bar, table cells and memory legend values. `tabular-nums` was removed from text
  (inputs, model facts, chips, verdicts, chart text, About code chips).
- **Seen:** in the breakdown table the share ("44%") touched its bound bar: `tbody td { padding: 0 }` overrode
  the column gap. **Changed:** the cell gap now comes after the reset (24 px, `--space-5`).
- **Seen:** the fit warning glyph floated at cap height between the figure and its unit ("3,953 ▲ tok/s @ 8k"),
  separating the number from its context. **Changed:** it follows the unit, at 16 px, sitting on the unit's
  baseline like a capital letter ("3,953 tok/s @ 8k ▲").
- **Seen (phone):** the masthead tagline wrapped the masthead to three lines. **Changed:** the tagline hides
  below 600 px; the masthead stays at 48 px.

## Pass 2: help popups, picker, states

- **Seen:** the Device "?" tooltip opened about 400 px below its icon. The `transform` used to nudge the icon
  made `.help` the containing block of the `position: fixed` popup. **Changed:** the nudge uses `top`, not a
  transform.
- **Seen:** the open (filled) "?" lost its circle: the fill sat at `z-index: -1` behind the pane background.
  **Changed:** the button is its own stacking context.
- **Seen:** long headlines broke after the "×" ("… on 2 ×" / "NVIDIA GeForce …"). **Changed:** the count, "×" and
  the device's first word never break apart.
- **Seen:** number fields read "0 . 75" (tabular full stop). **Changed:** no tabular figures in inputs.
- **Seen:** the picker popup, the help tooltip and the roofline tooltip render on white (field) surfaces with the
  blue-tinted shadow in light; on #17243B in dark. All three line up with their trigger. No change needed.

## Pass 3: zoomed crops (3×), the user's main complaint

- **Label + "?"** ("Weights ?", "KV cache ?", "Draft length k ?"): gap measured 20 device px = 6.7 CSS px
  (token 6 px plus glyph side bearing). Circle centre is 0.7 CSS px below the label's cap-height centre, which
  reads as centred on mixed-case labels; `--icon-nudge` stays 0. The 24 px hit area does not change the 20 px
  label line: label → control is 6 px on every field, field → next label 16 px.
- **Serif heading + "?"** ("Memory per device ?", "Bottleneck estimation ?"): the icon sat about 1 px low against
  Newsreader's tall x-height. **Changed:** it is lifted 1 px on output section heads only.
- **Verdict + "?"** ("Fits: 55.3GB of 60.0GB (92.2%) ?"), **chips + "?"**, **spec-dec + "?"**: on the text
  line, centred, one space plus 6 px gap. No change.
- **Picker trigger:** 36 px, 10 px text inset, chevron 7 px, 10 px from the right edge, vertically centred.
  Number fields, static fields, segmented controls and the Load button share the 36 px height (44 px on phones).
- **Legend swatches:** 10 px, centred on the x-height, 6 px to the text. No change.

## Pass 3b: the docs screenshot suite (`bun run screenshots`)

- **Seen (phone):** with a hypothetical drafter the decode line "(spec-dec ×2.16) ? hypothetical drafter" was
  set nowrap and pushed the page sideways. **Changed:** that line wraps.
- **Seen:** the "Memory per device" tooltip was set in Newsreader: the popup inherits the font of the serif
  heading that holds its "?". **Changed:** the popup and the "?" glyph are always in the grotesk.

## Pass 4: mini result bar, wide and mid widths

- **Seen (phone):** the mini bar appears at the top of the window once the figures scroll away and stays
  through the inputs. Values red when the model does not fit, since the fit text is hidden on phones. On desktop
  the values stay ink and the fit text ("Does not fit") carries the colour.
- **Seen (1024 × 768):** the output pane is narrower than 760 px there, so the figures go two-up and the
  roofline drops below them (container query). The splitter's 3 px blue hover line is visible.
- **Seen (1920 × 1080):** the output content stops at 1080 px; the inputs pane at 1/3 is 640 px, and paired
  fields sit two-up.
- **Above the fold at 1440 × 900 (default load):** headline, both figures, the roofline, the memory verdict and
  bar, and the top of the bottleneck table. Before: the table header sat at the bottom edge and the figures
  under a 80 px header.
