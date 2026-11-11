// Bundle-size budget, run after `bun run build`. Units: decimal kB (1000 bytes), as Vite prints them.
// Limits cover raw and gzip independently. Raise them deliberately in the same feature change, with a reason;
// see docs/deploy.md ("Bundle-size budget").
import { readdirSync, readFileSync } from 'node:fs';
import { gzipSync, brotliCompressSync } from 'node:zlib';

/** Limits per group. `total` includes lazy chunks, so deferral cannot hide an overall size regression. */
const LIMITS_KB = {
  main: { raw: 620, gzip: 190 },
  worker: { raw: 175, gzip: 52 },
  about: { raw: 595, gzip: 201 },
  total: { raw: 1890, gzip: 600 },
};

/** Group of a Vite chunk (`<name>-<hash>.js`). The sidecar is lazy but included in `total`. */
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
    return {
      file,
      group: groupOf(file),
      raw: bytes.length,
      gzip: gzipSync(bytes).length,
      brotli: brotliCompressSync(bytes).length,
    };
  })
  .sort((a, b) => b.raw - a.raw);

console.log('JS chunk sizes (kB, 1 kB = 1000 bytes):');
for (const r of rows)
  console.log(
    `  ${r.file.padEnd(52)} ${kb(r.raw).toFixed(2).padStart(9)} raw ${kb(r.gzip).toFixed(2).padStart(8)} gzip ${kb(r.brotli).toFixed(2).padStart(8)} brotli [${r.group}]`,
  );

let failed = false;
console.log('Budget (raw / gzip):');
for (const [group, limits] of Object.entries(LIMITS_KB)) {
  const members = group === 'total' ? rows : rows.filter((row) => row.group === group);
  if (members.length === 0) {
    console.error(`  ${group}: no chunk matches (was a chunk renamed? update groupOf)`);
    failed = true;
    continue;
  }
  for (const metric of ['raw', 'gzip']) {
    const size = kb(members.reduce((sum, row) => sum + row[metric], 0));
    const ok = size <= limits[metric];
    if (!ok) failed = true;
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${group.padEnd(7)} ${metric.padEnd(4)} ${size.toFixed(2).padStart(9)} / ${limits[metric]} kB`,
    );
  }
}
if (failed) {
  console.error(
    'Bundle-size budget exceeded. Shrink the chunk, or raise LIMITS_KB deliberately (docs/deploy.md).',
  );
  process.exit(1);
}
