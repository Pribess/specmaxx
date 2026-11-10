// Bundle-size budget: run after `bun run build`. Prints the raw and gzip size of every JS chunk in dist/assets/ and
// fails (exit 1) when a budgeted group exceeds its limit. Sizes are in kB = 1000 bytes, as Vite prints them.
//
// The limits sit about 10% above the sizes measured when they were last set. To raise one on purpose (a feature that
// really needs the bytes), edit LIMITS_KB below in the same change and say why in the commit message; see
// docs/deploy.md ("Bundle-size budget").
import { readdirSync, readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

/**
 * Raw (uncompressed) limits per group, in kB. `main`: the estimator's entry chunk; `worker`: the simulation worker;
 * `about`: the About page chunk (docs/simulation-model.md is bundled as text); `total`: every JS file in dist/assets.
 */
const LIMITS_KB = { main: 800, worker: 220, about: 595, total: 1890 };

/** The group of a chunk by its file name (`<name>-<hash>.js`; Vite's hash is appended to the entry name). */
function groupOf(file) {
  if (file.startsWith('main-')) return 'main';
  if (file.startsWith('sim.worker-')) return 'worker';
  if (file.startsWith('about-')) return 'about';
  return 'other';
}

const dir = 'dist/assets';
let files;
try {
  files = readdirSync(dir).filter((f) => f.endsWith('.js'));
} catch {
  console.error(`${dir} is missing: run \`bun run build\` first`);
  process.exit(1);
}

const kb = (bytes) => bytes / 1000;
const rows = files
  .map((file) => {
    const bytes = readFileSync(`${dir}/${file}`);
    return { file, group: groupOf(file), raw: bytes.length, gzip: gzipSync(bytes).length };
  })
  .sort((a, b) => b.raw - a.raw);

/** Raw size per budgeted group, bytes. */
const sizes = { main: 0, worker: 0, about: 0, total: 0 };
for (const r of rows) {
  if (r.group !== 'other') sizes[r.group] += r.raw;
  sizes.total += r.raw;
}

console.log('JS chunk sizes (kB, 1 kB = 1000 bytes):');
for (const r of rows)
  console.log(
    `  ${r.file.padEnd(44)} ${kb(r.raw).toFixed(2).padStart(9)} raw ${kb(r.gzip).toFixed(2).padStart(8)} gzip  [${r.group}]`,
  );

let failed = false;
console.log('Budget (raw):');
for (const [group, limit] of Object.entries(LIMITS_KB)) {
  if (group !== 'total' && !rows.some((r) => r.group === group)) {
    console.error(`  ${group}: no chunk matches (was a chunk renamed? update groupOf)`);
    failed = true;
    continue;
  }
  const size = kb(sizes[group]);
  const ok = size <= limit;
  if (!ok) failed = true;
  console.log(
    `  ${ok ? 'ok  ' : 'FAIL'} ${group.padEnd(7)} ${size.toFixed(2).padStart(9)} / ${limit} kB`,
  );
}
if (failed) {
  console.error(
    'Bundle-size budget exceeded. Shrink the chunk, or raise LIMITS_KB in scripts/check-bundle-size.mjs on purpose (docs/deploy.md).',
  );
  process.exit(1);
}
