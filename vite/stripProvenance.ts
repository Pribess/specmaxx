import ts from 'typescript';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';

const FULL = '?full-provenance';
const SIDECAR = 'virtual:hardware-provenance';
const FLAGS = 'virtual:community-unlocks';

/**
 * Empty citation expressions in preset object literals, without evaluating or rewriting physical data.
 * TypeScript's parser handles nested arrays, spreads and factory functions; dead citation constants are
 * subsequently tree-shaken. The pure data equivalent is `strip` in src/sim/hw/provenance.ts.
 * See docs/deploy.md ("Preset provenance sidecar").
 */
export function stripPresetCode(code: string, id: string): string {
  const file = ts.createSourceFile(id, code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const changes: { start: number; end: number; text: string }[] = [];
  const marked = new Set<ts.Node>();
  function visit(node: ts.Node): void {
    if (ts.isObjectLiteralExpression(node)) {
      const keys = node.properties
        .filter(
          (property) =>
            ts.isPropertyAssignment(property) || ts.isShorthandPropertyAssignment(property),
        )
        .map((property) => property.name!.getText(file));
      // Standalone SourceRef records also occur in private helpers and SFU-rate constants. Empty
      // those records too: conservative tree shaking can otherwise retain their unused payloads.
      if (keys.includes('url') && keys.includes('title')) {
        changes.push({ start: node.getStart(file), end: node.end, text: '({})' });
        return;
      }
    }
    if (ts.isPropertyAssignment(node) || ts.isShorthandPropertyAssignment(node)) {
      const key = node.name.getText(file).replace(/^['"]|['"]$/g, '');
      if (key === 'reason') {
        changes.push({ start: node.getStart(file), end: node.end, text: "reason: ''" });
        return;
      }
      if (key === 'sources' || key === 'estimated' || key === 'estimatedReason') {
        const marker = marked.has(node.parent) ? '' : ', provenanceStripped: true';
        marked.add(node.parent);
        changes.push({
          start: node.getStart(file),
          end: node.end,
          text: `${key}: ${key === 'estimatedReason' ? 'undefined' : '[]'}${marker}`,
        });
        return;
      }
    }
    // Catalog report(...) arguments sometimes copy another preset's first source URL. Their result
    // is discarded too, but evaluating the argument must not index an emptied citation array.
    if (
      ts.isElementAccessExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      ['sources', 'estimated'].includes(node.expression.name.text)
    ) {
      changes.push({ start: node.getStart(file), end: node.end, text: '({})' });
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(file);
  for (const change of changes.sort((a, b) => b.start - a.start))
    code = code.slice(0, change.start) + change.text + code.slice(change.end);
  return code;
}

/**
 * Build-only stripping for both Vite graphs (main and worker). The explicitly requested full graph is
 * reachable solely through a dynamic import; its query propagates through preset imports, so factories
 * retain the original citations too. Dev and Vitest serve unchanged source presets.
 */
export function stripProvenance(): Plugin {
  let building = false;
  let root = '';
  return {
    name: 'strip-preset-provenance',
    enforce: 'pre',
    configResolved(config) {
      building = config.command === 'build';
      root = config.root;
    },
    async resolveId(source, importer) {
      if (source === SIDECAR || source === FLAGS) return `\0${source}`;
      if (source.endsWith(FULL) || importer?.endsWith(FULL)) {
        const bare = source.replace(FULL, '');
        const parent = importer?.replace(FULL, '');
        const resolved = await this.resolve(bare, parent, { skipSelf: true });
        if (!resolved) return null;
        // Only hardware.ts and presets need a second graph. Shared pure helpers stay deduplicated.
        if (resolved.id.includes('/sim/hw/presets/') || source.endsWith(FULL))
          return `${resolved.id}${FULL}`;
        return resolved;
      }
      return null;
    },
    async load(id) {
      if (id === `\0${SIDECAR}`) {
        return `export { resolveHardware } from ${JSON.stringify(resolve(root, 'src/ui/state/hardware.ts') + FULL)};
export { DEVICE_UNLOCKS } from ${JSON.stringify(resolve(root, 'src/sim/hw/presets/index.ts') + FULL)};`;
      }
      if (id === `\0${FLAGS}`) {
        // Dev imports live source modules so Vite tracks preset changes for HMR. Production evaluates
        // originals through Vite only when generating this module, never while loading vite.config.ts.
        if (!building)
          return `import { DEVICE_UNLOCKS } from '/src/sim/hw/presets/index.ts';
export const COMMUNITY_UNLOCKS = new Set(Object.entries(DEVICE_UNLOCKS).flatMap(([device, axes]) =>
  axes.flatMap(axis => axis.options.filter(option =>
    [...option.deviceModifiers, ...(option.intraNodeLinkModifiers ?? [])].some(mod =>
      mod.sources.some(source => source.kind === 'community'))
  ).map(option => device + '/' + axis.id + '/' + option.id))));`;
        const { createServer } = await import('vite');
        const server = await createServer({
          configFile: false,
          root,
          optimizeDeps: { noDiscovery: true, include: [] },
          server: { middlewareMode: true, watch: null, hmr: false },
          appType: 'custom',
        });
        let DEVICE_UNLOCKS: typeof import('../src/sim/hw/presets/index.ts').DEVICE_UNLOCKS;
        try {
          ({ DEVICE_UNLOCKS } = await server.ssrLoadModule('/src/sim/hw/presets/index.ts'));
        } finally {
          await server.close();
        }
        const community: string[] = [];
        for (const [device, axes] of Object.entries(DEVICE_UNLOCKS))
          for (const axis of axes)
            for (const option of axis.options)
              if (
                [...option.deviceModifiers, ...(option.intraNodeLinkModifiers ?? [])].some((mod) =>
                  mod.sources.some((source) => source.kind === 'community'),
                )
              )
                community.push(`${device}/${axis.id}/${option.id}`);
        return `export const COMMUNITY_UNLOCKS = new Set(${JSON.stringify(community)});`;
      }
      if (id.endsWith(FULL)) {
        // Full sources are separate module IDs, never a runtime request or a public static file.
        const { readFile } = await import('node:fs/promises');
        return readFile(id.slice(0, -FULL.length), 'utf8');
      }
      return null;
    },
    transform(code, id) {
      if (!building || id.includes('?') || !id.includes('/sim/hw/presets/')) return null;
      // Deliberately map the generated preset module, not deleted citation blocks. Vite retains
      // per-module source-map attribution, but original preset line/column breakpoints are unavailable
      // in production. Dev keeps unchanged sources and exact locations; no extra mapping dependency.
      return { code: stripPresetCode(code, id), map: null };
    },
  };
}
