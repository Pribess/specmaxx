/**
 * The body of a "?" help popup: one sentence per line, each on its own visible line.
 * Pass the lines as an array, or as one string with `\n` between lines.
 */
export function helpLines(lines: readonly string[] | string) {
  const list = typeof lines === 'string' ? lines.split('\n') : lines;
  return list.map((line) => (
    <span key={line} className="help-line">
      {line}
    </span>
  ));
}
