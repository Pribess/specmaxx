// Bundle-size budget, run after `bun run build`. Units: decimal kB (1000 bytes), as Vite prints them.
// Limits cover raw and gzip independently. Raise them deliberately in the same feature change, with a reason;
// see docs/deploy.md ("Bundle-size budget").
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { gzipSync, brotliCompressSync } from 'node:zlib';

/**
 * Limits per group. `total` includes lazy chunks and the lazily fetched About documents, so deferral cannot hide an
 * overall size regression. The `doc:*` groups are the reference documents the About page fetches on demand (prerendered
 * HTML, see vite/prerenderDocs.ts); their limits are their size plus about 10%.
 */
const LIMITS_KB = {
  main: { raw: 620, gzip: 190 },
  worker: { raw: 175, gzip: 52 },
  about: { raw: 100, gzip: 35 },
  'doc:simulation-model': { raw: 446, gzip: 153 },
  'doc:sanity-check': { raw: 56, gzip: 19 },
  total: { raw: 1890, gzip: 600 },
};

/** Groups whose source is optional: docs/sanity-check.md is not always in the repository. */
const PRESENT = { 'doc:sanity-check': existsSync('docs/sanity-check.md') };

/**
 * Group of an emitted file (`<name>-<hash>.js`, or `<name>-<hash>.htm` for a lazy About document). The sidecar and
 * shared chunks are `other`, included in `total` only.
 */
function groupOf(file) {
  if (file.startsWith('main-')) return 'main';
  if (file.startsWith('sim.worker-')) return 'worker';
  if (file.startsWith('about-') && file.endsWith('.js')) return 'about';
  const doc = /^(simulation-model|sanity-check)-[A-Za-z0-9_-]{8}\.htm$/.exec(file);
  if (doc) return `doc:${doc[1]}`;
  return 'other';
}

const dir = 'dist/assets';
let files;
try {
  files = readdirSync(dir).filter((f) => f.endsWith('.js') || f.endsWith('.htm'));
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

console.log('JS chunk and lazy document sizes (kB, 1 kB = 1000 bytes):');
for (const r of rows)
  console.log(
    `  ${r.file.padEnd(52)} ${kb(r.raw).toFixed(2).padStart(9)} raw ${kb(r.gzip).toFixed(2).padStart(8)} gzip ${kb(r.brotli).toFixed(2).padStart(8)} brotli [${r.group}]`,
  );

let failed = false;
console.log('Budget (raw / gzip):');
for (const [group, limits] of Object.entries(LIMITS_KB)) {
  const members = group === 'total' ? rows : rows.filter((row) => row.group === group);
  if (members.length === 0) {
    if (PRESENT[group] === false) {
      console.log(`  skip ${group}: its source document is not in the repository`);
      continue;
    }
    console.error(`  ${group}: no file matches (was a file renamed? update groupOf)`);
    failed = true;
    continue;
  }
  for (const metric of ['raw', 'gzip']) {
    const size = kb(members.reduce((sum, row) => sum + row[metric], 0));
    const ok = size <= limits[metric];
    if (!ok) failed = true;
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${group.padEnd(20)} ${metric.padEnd(4)} ${size.toFixed(2).padStart(9)} / ${limits[metric]} kB`,
    );
  }
}
if (failed) {
  console.error(
    'Bundle-size budget exceeded. Shrink the chunk, or raise LIMITS_KB deliberately (docs/deploy.md).',
  );
  process.exit(1);
}
