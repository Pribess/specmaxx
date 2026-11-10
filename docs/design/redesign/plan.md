# specmaxx redesign: design plan

Status: plan for review (phase 1). Nothing in `src/` has changed yet.
Before shots: `docs/design/redesign/before/`. Research notes on BBC and Reuters (with sources) were
gathered for this plan. The traits below are the ones adopted from them.

## 1. Subject, audience, job

- **Subject:** the theoretical maximum prefill and decode throughput of an LLM on a hardware stack.
- **Audience:** engineers sizing hardware. They read numbers closely, compare configurations, and
  change one input at a time.
- **Primary job:** answer "how fast can this model run on this rig, at what context, and what limits
  it?". The answer is two numbers with their context ("3,953 tok/s @ 8k"), a fit verdict and a bound.

The news analogy is literal. Each configuration is a _story_: a headline ("Qwen3.8-27B on 1 × RTX
5090, TP 1"), two lead figures, then the analysis (memory, bottlenecks, sweep). That is how
BBC and Reuters data pages are built, and the layout follows it.

## 2. Traits adopted from BBC and Reuters

| Trait                                                                                                                                                                        | Source                          | How we use it                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Serif headline + sans for everything else (BBC Reith Serif / Reith Sans; Reuters Graphics Newsreader + Knowledge)                                                            | BBC live site, Reuters Graphics | Newsreader for the configuration headline and section heads; Schibsted Grotesk for labels, values, controls and tables |
| Short type scale: 16/24 body, 12–14 px meta, headline 28–38 px with slight negative tracking (BBC −0.02em)                                                                   | BBC GEL, Reuters tokens         | Scale in §5; headline −0.015em                                                                                         |
| One big figure per data card (Reuters number card)                                                                                                                           | Reuters                         | The two hero figures, tabular, at 64 px                                                                                |
| Labels small and grey, values dark and heavier; tables with tabular cells, a ink-coloured rule under the header and light rules between rows                                 | Reuters                         | Label/value contract in §5; breakdown table                                                                            |
| Separation mostly by space, with grey fills for control areas and panels, no heavy boxes                                                                                     | BBC, Reuters                    | Inputs pane on a tinted surface; output sections separated by space                                                    |
| Accent colour only on small elements (links, focus, selected state); links not marked by colour alone                                                                        | both                            | Deep blue for masthead, focus, selection, links (underlined)                                                           |
| Solid 2 px focus outline                                                                                                                                                     | Reuters                         | `--focus-ring`                                                                                                         |
| Charts without axis lines or ticks, only light horizontal gridlines; emphasised baseline; 1 px hover cursor; tooltip = white box, 1 px border, 8 px padding, bold first line | BBC bbplot, Reuters             | Roofline and sweep charts                                                                                              |
| Measure ≤ 65 ch for prose; 8 px baseline; more space above a heading than below                                                                                              | BBC GEL                         | About page measure 64 ch; spacing scale on 4/8                                                                         |

**Not adopted, deliberately:** their brand colours (BBC black/red, Reuters orange), and the small
**uppercase tracked section labels** both sites use. Tracked all-caps labels are one of the
commonest generated-page tells; sentence-case semibold labels in a muted blue-grey carry the same
hierarchy. Also not adopted: the "broadsheet" look (hairline rules everywhere, zero radius, dense
columns); see §10.

## 3. Palette (deep blue)

Primary is a deep, slightly cool ultramarine ("Admiralty"), not navy-black and not a SaaS azure.
Greys are tuned toward it (hue ≈ 215°, low chroma). Danger moves to a crimson that sits next to the
blue without vibrating; warning becomes a burnt orange of the same depth.

### 3.1 Scales (shared by both schemes)

**Admiralty (primary)**

| Token        | Hex           | Use                                                           |
| ------------ | ------------- | ------------------------------------------------------------- |
| blue-50      | `#EEF3FB`     | selected row tint (light)                                     |
| blue-100     | `#DCE6F6`     | hover tint, selected option (light)                           |
| blue-200     | `#B9CCEC`     | masthead secondary text, focus halo                           |
| blue-300     | `#8FB4F0`     | primary on dark (links, focus, selection outline)             |
| blue-400     | `#5A82C6`     | splitter hover (dark)                                         |
| blue-500     | `#2F5EAD`     | selected segment fill (dark)                                  |
| blue-600     | `#1A4A99`     | hover of primary fill (light)                                 |
| **blue-700** | **`#0F3B82`** | **primary: masthead (light), selected segment, links, focus** |
| blue-800     | `#0B2D66`     | masthead (dark)                                               |
| blue-900     | `#08204A`     | pressed                                                       |
| blue-950     | `#051430`     | —                                                             |

**Slate (blue-tuned greys)**

| Token     | Hex                                  |
| --------- | ------------------------------------ |
| slate-0   | `#FFFFFF` (field fill, light)        |
| slate-25  | `#F7F9FC` (paper, light)             |
| slate-50  | `#EDF1F7` (mist: inputs pane, light) |
| slate-100 | `#E2E8F1` (track, light)             |
| slate-200 | `#C9D2DE` (rules, gridlines, light)  |
| slate-400 | `#7D8AA0` (control borders, light)   |
| slate-500 | `#55627A` (muted text, light)        |
| slate-600 | `#9DABC0` (muted text, dark)         |
| slate-700 | `#6A7A93` (control borders, dark)    |
| slate-800 | `#2B3A55` (rules, dark)              |
| slate-850 | `#17243B` (field fill, dark)         |
| slate-900 | `#111C30` (inputs pane, dark)        |
| slate-950 | `#0B1424` (paper, dark)              |
| ink       | `#14213A` (text, light)              |
| frost     | `#E6ECF5` (text, dark)               |

**Semantic and regime colours** (regimes encode bounds only, as today)

| Token                  | Light                                        | Dark      |
| ---------------------- | -------------------------------------------- | --------- |
| danger (crimson)       | `#B4232F`                                    | `#FF8A80` |
| warning (burnt orange) | `#B54708`                                    | `#F5A35C` |
| compute bound (violet) | `#6A3DC2`                                    | `#A88BF0` |
| bandwidth bound (teal) | `#0A6E77` (was `#0B7A83`, failed AA on mist) | `#47C1C9` |
| comm (ochre)           | `#8F5B00` (was `#A86A00`, 4.21 failed AA)    | `#F0B44C` |

### 3.2 Roles

| Role token                                           | Light                             | Dark                              |
| ---------------------------------------------------- | --------------------------------- | --------------------------------- |
| `--paper` (output pane, page)                        | slate-25 `#F7F9FC`                | slate-950 `#0B1424`               |
| `--mist` (inputs pane)                               | slate-50 `#EDF1F7`                | slate-900 `#111C30`               |
| `--field` (input/picker fill)                        | `#FFFFFF`                         | slate-850 `#17243B`               |
| `--ink` (text)                                       | `#14213A`                         | `#E6ECF5`                         |
| `--muted` (labels, units, meta)                      | `#55627A`                         | `#9DABC0`                         |
| `--rule` (gridlines, table rules, splitter)          | `#C9D2DE`                         | `#2B3A55`                         |
| `--control-border`                                   | `#7D8AA0`                         | `#6A7A93`                         |
| `--primary` (links, focus, selection)                | `#0F3B82`                         | `#8FB4F0`                         |
| `--primary-fill` / `--on-primary` (selected segment) | `#0F3B82` / `#FFFFFF`             | `#2F5EAD` / `#FFFFFF`             |
| `--masthead` / `--on-masthead` / `--masthead-muted`  | `#0F3B82` / `#FFFFFF` / `#B9CCEC` | `#0B2D66` / `#E6ECF5` / `#B9CCEC` |
| `--tint` (hover, selected option)                    | blue-100 `#DCE6F6`                | `#1B2C4A`                         |

### 3.3 WCAG contrast (computed, WCAG 2.x relative luminance)

| Pair                                    | Light       | Dark  | AA  |
| --------------------------------------- | ----------- | ----- | --- |
| ink on paper                            | 15.22       | 15.51 | ✓   |
| ink on mist                             | 14.16       | 14.34 | ✓   |
| ink on field                            | 16.05       | 13.08 | ✓   |
| muted on paper                          | 5.83        | 7.91  | ✓   |
| muted on mist                           | 5.43        | 7.32  | ✓   |
| muted on field                          | 6.15        | 6.67  | ✓   |
| primary (link) on paper                 | 10.12       | 8.73  | ✓   |
| primary on mist                         | 9.41        | 8.08  | ✓   |
| primary on tint                         | 8.48        | —     | ✓   |
| on-primary on primary-fill              | 10.67       | 6.32  | ✓   |
| on-masthead on masthead                 | 10.67       | 11.18 | ✓   |
| masthead-muted on masthead              | 6.33        | 7.88  | ✓   |
| danger on paper                         | 6.18        | 8.07  | ✓   |
| danger on mist                          | 5.75        | 7.46  | ✓   |
| warning on paper                        | 5.15        | 9.02  | ✓   |
| warning on mist                         | 4.79        | 8.34  | ✓   |
| compute (violet) on paper               | 6.51        | 6.68  | ✓   |
| bandwidth (teal) on paper / mist        | 5.67 / 5.28 | 8.55  | ✓   |
| comm (ochre) on paper / mist            | 5.43 / 5.06 | 9.95  | ✓   |
| control border vs field (non-text, ≥ 3) | 3.49        | 3.56  | ✓   |
| control border vs mist (non-text, ≥ 3)  | 3.08        | 3.91  | ✓   |

Rules (`--rule`) are decorative separators, not control boundaries, so they sit below 3:1 on purpose.

## 4. Typefaces

| Family                                                                        | Package (pinned)                                           | Role                                                                                                                                | Figures                                                                                             |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Newsreader** (Production Type, designed for news; Reuters Graphics' serif)  | `@fontsource-variable/newsreader@5.3.0` (opsz + wght axes) | Configuration headline, section heads (Memory per device, Bottleneck estimation, input section heads), About page headings and body | digits tabular by default                                                                           |
| **Schibsted Grotesk** (built for the Schibsted news group; normal width only) | `@fontsource-variable/schibsted-grotesk@5.3.0`             | Wordmark, labels, values, hero figures, controls, tables, chart text, tooltips                                                      | `font-variant-numeric: tabular-nums` (has `tnum`) on every value, table cell, axis label and figure |

- Archivo (`@fontsource-variable/archivo`) is removed. Its `wdth` axis goes with it, so no narrow
  widths can appear; neither new face has a width axis.
- Two families, clearly distinct: a news serif for the story voice, a grotesk for data. No
  monospace anywhere (op names in the table use the grotesk too).
- Hero figures are in the grotesk, not the serif: at 64 px the grotesk's flat-topped tabular digits
  line up across Prefill and Decode, and serif oldstyle-like contrast hurts at a glance.

## 5. Type scale

8 px baseline; line heights land on multiples of 4. Sizes in px (rem in code, 1 rem = 16 px).

| Token             | Size / line                       | Face, weight              | Use                                             |
| ----------------- | --------------------------------- | ------------------------- | ----------------------------------------------- |
| `--type-figure`   | 64 / 64, −0.02em (phone 44 / 48)  | Grotesk 750, tnum         | Prefill and decode figures                      |
| `--type-headline` | 30 / 36, −0.015em (phone 22 / 28) | Newsreader 560, opsz auto | Configuration headline                          |
| `--type-section`  | 22 / 28                           | Newsreader 600            | Output section heads                            |
| `--type-group`    | 18 / 24                           | Newsreader 600            | Input section heads (Model, Quantization, …)    |
| `--type-body`     | 16 / 24                           | Grotesk 400               | Body, About page lists                          |
| `--type-value-lg` | 18 / 24                           | Grotesk 650, tnum         | Figure-adjacent values (memory verdict)         |
| `--type-control`  | 15 / 20                           | Grotesk 450               | Picker and field text                           |
| `--type-label`    | 14 / 20                           | Grotesk 600               | Field labels                                    |
| `--type-unit`     | 16 / 20                           | Grotesk 500, muted        | "tok/s @ 8k" next to figures                    |
| `--type-meta`     | 13 / 16                           | Grotesk 450, muted        | Units, legends, table headers, axis text, hints |
| `--type-micro`    | 12 / 16                           | Grotesk 500, muted        | Chart point labels only                         |

Label vs value contract: **labels** are muted, semibold 14 px, sentence case; **values** are ink,
tabular, one step heavier than the surrounding text, units muted and one step smaller, separated by
a thin space. Every throughput number keeps its context ("tok/s @ 8k").

## 6. Spacing and control sizes

### 6.1 Spacing (4 px base)

`--space-1` 4 · `--space-2` 8 · `--space-3` 12 · `--space-4` 16 · `--space-5` 24 · `--space-6` 32 ·
`--space-7` 48 · `--space-8` 64

| Gap                                        | Token                 | Value                                                    |
| ------------------------------------------ | --------------------- | -------------------------------------------------------- |
| label ↔ "?" icon                           | `--gap-label-icon`    | 6 px                                                     |
| icon ↔ text (warning glyph, legend swatch) | `--gap-icon-text`     | 6 px                                                     |
| label row ↔ control                        | `--gap-label-control` | 6 px                                                     |
| field ↔ field (vertical)                   | `--gap-field`         | 16 px                                                    |
| fields side by side                        | `--gap-field-inline`  | 12 px                                                    |
| input section ↔ section                    | `--gap-group`         | 32 px                                                    |
| output section ↔ section                   | `--gap-section`       | 48 px                                                    |
| heading ↔ its content                      | `--gap-heading`       | 12 px (space above a heading is always ≥ 2× space below) |
| pane padding                               | `--pane-pad`          | 32 px desktop, 16 px phone                               |

### 6.2 Control sizes

| Token                                                                       | Desktop      | Phone (coarse pointer) |
| --------------------------------------------------------------------------- | ------------ | ---------------------- |
| `--control-h` (picker trigger, number field, text field, segmented, button) | 36 px        | 44 px                  |
| `--control-h-sm` (table/chart toolbar segmented)                            | 28 px        | 36 px                  |
| `--control-pad-x`                                                           | 10 px        | 12 px                  |
| `--control-radius`                                                          | 4 px         | 4 px                   |
| `--popover-radius`                                                          | 6 px         | 6 px                   |
| `--icon-help` (visible "?" circle)                                          | 15 px        | 17 px                  |
| `--icon-help-hit` (invisible hit area, negative margin)                     | 24 px        | 32 px                  |
| `--icon-inline` (warning glyph, chevron)                                    | 14 px        | 16 px                  |
| `--swatch` (legend)                                                         | 10 px        | 10 px                  |
| `--splitter-line` / `--splitter-hit`                                        | 1 px / 12 px | —                      |

Radius by hierarchy, not one radius for everything: 4 px controls, 6 px popovers and tooltips,
0 on panes, full round only on the "?" icon and the scrollbar thumb.

### 6.3 Inline icon alignment

Label rows are `display: flex; align-items: center; gap: var(--gap-label-icon)` with a fixed
20 px line box. The "?" glyph is a 15 px circle centred in a 24 px hit box whose margins are
negative, so the hit area does not push the label row taller than 20 px or shift the next field.
Optical target: the circle's centre sits on the label's **cap-height centre** (≈ 0.36 em above the
baseline), not on the line box centre; the residual offset is fixed with a token
(`--icon-nudge`, expected ≈ −0.5 px) and verified on zoomed crops of rendered labels. The same rule
holds for the warning glyph beside figures and verdicts, and for legend swatches.

## 7. Layout

### 7.1 Desktop (≥ 900 px): one container, output left 2/3, inputs right 1/3

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ specmaxx /̄   Theoretical ceilings for your rig                    How it works   │ masthead 56px, deep blue
├───────────────────────────────────────────────────────┬┬─────────────────────────┤
│ OUTPUT (paper, ScrollArea)                     ▲      ││ INPUTS (mist, ScrollArea)│
│                                                │      ││                         │
│ Qwen/Qwen3.8-27B on 1 × NVIDIA GeForce              ││ Model                   │
│ RTX 5090, TP 1                    (serif headline)  ││ Hugging Face model ID ? │
│                                                     ││ [Qwen/Qwen3.8-27B ][Load]│
│ Prefill              Decode                         ││ 26.9B total · 64 layers │
│ 3,953 ▲ tok/s @ 8k   34.4 ▲ tok/s @ 8k              ││                         │
│                                                     ││ Quantization            │
│ ▲ Does not fit: 55.3 GB of 32.0 GB  [■■■■■■■|■■■]   ││ Weights ?               │
│ Read how the estimate works                         ││ [Model default (bf16) v]│
│                                                     ││ Activations ? KV cache ?│
│ Roofline                    │ Memory per device ?   ││ [bf16        v][bf16  v]│
│  100 ┤      ____ Compute    │ Weights 53.8 GB       ││                         │
│   10 ┤    /                 │ KV + state 691 MB     ││ Hardware                │
│    1 ┤  o  Bandwidth        │ Activations 856 MB    ││ Device ?                │
│      1   10   100  1,000    │ max 0 seq @ 8k        ││ [NVIDIA GeForce …    v] │
│                                                     ││ GPUs in one machine ?   │
│ Bottleneck estimation ?   [Decode@8k|Prefill] [Op|…]││ [ 1 | 2 | 4 | 8 ]       │
│ ■compute ■memory ▨comm latency ■comm bandwidth      ││ …                       │
│ Operation        Time    Share  Bound               ││                         │
│ mlp.gate_up     12.7 ms   44 %  ▬▬▬▬ memory         ││ Parallelism / Workload  │
│ …                                                   ││ Assumptions             │
│ Batch sweep (chart)                                 ▼││                        ▼│
└───────────────────────────────────────────────────────┴┴─────────────────────────┘
                                                        ↑ splitter: 1px rule, 12px hit
```

- The workspace is exactly the viewport height below the masthead (`100dvh − 56px`); the
  document itself does not scroll. Each pane is a themed ScrollArea and scrolls on its own, so the
  hero figures stay in view while any input is edited.
- Output pane: content left-aligned, max line length 72 ch for any text; the figures, verdict and
  roofline are the first screen at 1440 × 900. When the hero scrolls out of the output pane, the
  **mini result bar** (§7.3) pins to the top of the output pane.
- Inputs pane: one column. Short paired inputs (Activations + KV cache, From layer + Resident %,
  Network link + Cabling) sit two-up when the pane is ≥ 400 px wide (container query), else stack.
- Alignment: everything left-aligned to the pane padding; numeric table columns right-aligned;
  nothing centred except the "?" glyph.

### 7.2 Splitter

| Property     | Spec                                                                                                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Visual       | 1 px `--rule` line at rest; on hover/focus/drag a 3 px `--primary` line with `col-resize` cursor; 12 px transparent hit area                                                                                                                            |
| Measure      | inputs pane width as a fraction of the workspace; default **1/3**                                                                                                                                                                                       |
| Limits       | inputs pane ≥ 300 px and ≥ 22 %; ≤ 50 %; output pane ≥ 480 px (clamped again on window resize)                                                                                                                                                          |
| Pointer      | pointer capture on drag; text selection disabled during drag                                                                                                                                                                                            |
| Keyboard     | focusable (`tabIndex=0`); ← / → move 2 % (Shift: 10 %); Home / End jump to min / max; Enter resets                                                                                                                                                      |
| Reset        | double-click (and Enter) resets to 1/3 and clears the stored value                                                                                                                                                                                      |
| ARIA         | `role="separator"`, `aria-orientation="vertical"`, `aria-label="Resize inputs"`, `aria-controls` = inputs pane id, `aria-valuenow` / `aria-valuemin` / `aria-valuemax` as whole percentages of the inputs pane, `aria-valuetext` "Inputs 33 % of width" |
| Persistence  | `localStorage['specmaxx:split']` = fraction (0–1); invalid or missing → default; storage errors ignored                                                                                                                                                 |
| Help         | a "?" is not needed (standard control); its `title` reads "Drag to resize · double-click to reset"                                                                                                                                                      |
| Below 900 px | not rendered                                                                                                                                                                                                                                            |

### 7.3 Phone (< 900 px): one column, stacked

```
┌──────────────────────────────┐
│ specmaxx /̄       How it works│ masthead 48px
├──────────────────────────────┤
│ Qwen/Qwen3.8-27B on 1 ×      │ 1 headline (serif 22)
│ RTX 5090, TP 1               │
│ Prefill        Decode        │ 2 figures, two-up, 44px
│ 3,953 ▲        34.4 ▲        │
│ tok/s @ 8k     tok/s @ 8k    │
│ ▲ Does not fit: 55.3/32.0 GB │ 3 memory verdict + bar
│ [■■■■■■■■■■■|■■■■■]          │
├──────────────────────────────┤ ← sticky mini bar once 1–3 scroll away:
│ 3,953 · 34.4 tok/s @ 8k  ▲   │   "Prefill 3,953 | Decode 34.4 tok/s @ 8k" + fit glyph
├──────────────────────────────┤
│ Model / Quantization /       │ 4 inputs (all sections, mist surface,
│ Hardware / Parallelism /     │   full bleed)
│ Workload / Assumptions       │
├──────────────────────────────┤
│ Roofline chart               │ 5 analysis: roofline, memory detail,
│ Memory per device            │   bottleneck table, batch sweep, notes
│ Bottleneck estimation        │
│ Batch sweep                  │
└──────────────────────────────┘
```

Order: **headline → figures → memory verdict → inputs → analysis**, with a sticky mini result
bar. Why: on a phone the loop is "change one input, see the number". Putting the answer first
answers the visit's question at once; inputs come next because editing is the main action and
they are long; the analysis is read once a configuration settles. The mini bar (36 px, mist
surface, tabular numbers, the fit glyph in danger/warning colour) keeps the two figures in view
during the whole input section, so the edit-and-see loop works without scrolling back. It is the
same component the desktop output pane pins. Document scroll on phone (one native, themed page
scrollbar, per the project rules).

## 8. Components (the spacing/size pass)

- **Masthead:** deep-blue band; wordmark in Grotesk 800 with the existing roof mark in
  `--on-masthead`; tagline in `--masthead-muted`, 14 px; "How it works" underlined.
- **Field:** label row (20 px) → 6 px → control (36 px). Number fields and pickers share height,
  padding, border, radius and font; the chevron is 14 px, 10 px from the right edge.
- **Segmented:** same height as a control; selected = primary fill; unselected = field fill;
  1 px control border around the group, no inner double border.
- **Disclosures (Advanced, input sections):** chevron 14 px before the text, 6 px gap, same left
  edge as labels; the whole row is the hit target.
- **Help icon:** §6.3. Tooltip: field fill, 1 px `--rule` border, 6 px radius, 8 × 12 px padding,
  13/18 text, blue-tinted shadow `0 6px 20px rgb(8 32 74 / 0.16)`.
- **Picker popover:** search box fixed on top (36 px), list in ScrollArea, options 32 px, active
  option `--tint` with a 2 px primary bar on the left edge, matched characters semibold.
- **Memory bar:** 10 px track, segments with the existing patterns; overflow in danger; legend
  swatches 10 px, 6 px to text, 16 px between items, values tabular.
- **Breakdown table:** header 13 px muted with a 1 px ink rule under it; rows 32 px with 1 px
  `--rule` between; numeric columns right-aligned tabular; bound bars 6 px high.
- **Charts:** no axis lines or ticks; light horizontal gridlines; 13 px muted axis labels; point
  labels 12 px; tooltip per the help tooltip style, bold first line, units on every value.
- **About page:** same masthead; Newsreader 18/30 body at 64 ch measure, headings in Newsreader,
  code in the grotesk on a `--mist` chip (no monospace).

## 9. Principles

1. **The figure is the headline's proof.** One bold place: the serif headline and the two
   64 px tabular figures. Everything else is quiet.
2. **Labels whisper, values speak.** Muted semibold labels, ink tabular values, muted units.
   Never a number without its unit and context.
3. **Tone and space separate; rules only where the data has rows.** Panes differ by surface
   (paper vs mist); sections by space; rules appear in tables and chart gridlines only.
4. **Deep blue means "you can act here or you are here".** Masthead, links, focus, selection,
   splitter. Never decoration, never a background wash.
5. **Every inline icon sits on the text it belongs to.** Fixed line boxes, tokens for gaps, checked
   on zoomed renders, not trusted from code.
6. **No prose.** Definitions live in "?" icons; visible text is labels, values and links.

## 10. Review against the brief and the skill's generated defaults

| Default / risk                                                                               | Verdict                                                                                                    | Change made                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Cream background + high-contrast serif + terracotta                                       | Avoided: cool paper `#F7F9FC`, deep-blue accent, Newsreader at moderate contrast                           | —                                                                                                                                                                                                                         |
| 2. Near-black + single acid accent                                                           | Avoided: dark scheme is deep navy `#0B1424` with a light-blue primary; regimes keep their own hues         | —                                                                                                                                                                                                                         |
| 3. Broadsheet hairline rules, zero radius, dense columns                                     | **Risk** (news brief invites it)                                                                           | First draft separated every input section and output section with 1 px rules and set 0 radius. Revised: panes separate by surface tone, sections by space; rules only in tables and gridlines; 4/6 px radius by hierarchy |
| 4. SaaS-card kit (identical rounded cards, grey shadows, gradients)                          | Avoided: no cards; one shadow, only on floating layers, tinted to the blue                                 | —                                                                                                                                                                                                                         |
| 5a. Tracked ALL-CAPS eyebrows                                                                | **Risk** (both BBC and Reuters use them)                                                                   | First draft had uppercase tracked section labels. Revised to sentence-case serif heads; labels sentence case                                                                                                              |
| 5b. Meta joined with middle dots                                                             | Present today in the compact results ("Prefill … · Decode …")                                              | Revised: the mini bar uses two labelled cells, no dot separator                                                                                                                                                           |
| 5c. Tinted near-black standing in for black                                                  | Considered: ink `#14213A` is a real blue ink derived from the primary (hue 220°), not a neutral #111       | Kept, as a palette choice                                                                                                                                                                                                 |
| 5d. Monospace for small data labels                                                          | Avoided: no monospace (op names and About code chips use the grotesk)                                      | —                                                                                                                                                                                                                         |
| 5e. "→" on links                                                                             | Avoided                                                                                                    | —                                                                                                                                                                                                                         |
| Big number + small label + gradient accent hero                                              | The figures _are_ the subject's answer, so a big number is right here; no gradient, no stat row decoration | Hero kept typographic: serif headline above, figures with muted units                                                                                                                                                     |
| Kicker bars above headings                                                                   | First draft added a 32 × 3 px blue bar above output section heads                                          | Removed: decoration, not information                                                                                                                                                                                      |
| Brief: inputs right 1/3, splitter, independent scrolling                                     | Covered in §7.1–7.2                                                                                        | —                                                                                                                                                                                                                         |
| Brief: phone order justified, sticky mini bar                                                | §7.3                                                                                                       | —                                                                                                                                                                                                                         |
| Brief: spacing and size tokens, icon baseline                                                | §6                                                                                                         | —                                                                                                                                                                                                                         |
| Project rules: no native choosers or scrollbars, "?" icons, chart tooltips, no narrow widths | Kept; Archivo's width axis removed entirely                                                                | —                                                                                                                                                                                                                         |

## 11. As built (differences from this plan, decided from renders)

- **Desktop hero:** the figures stack in a left column with the roofline beside them, and memory runs full
  width below (the wireframe showed figures two-up and roofline | memory). Stacking keeps both figures and the
  chart above the fold at 1440 × 900 with the bottleneck table starting on the first screen. When the output
  pane is narrower than 760 px (container query) the figures go two-up and the roofline drops below them.
- **Mini result bar:** 44 px, not 36 px. It holds the label, the 18 px value and the unit on one baseline.
- **Tabular figures:** Schibsted Grotesk's `tnum` also widens `,` `.` `:`, so numbers go through `Num`
  (digits tabular, separators proportional), and text never gets `tabular-nums` (see notes.md, pass 1).
- **Help icon nudge:** `--icon-nudge` stays 0 on labels (0.7 px off the cap-height centre, measured on 3×
  crops); serif section heads lift their icon 1 px.
