/**
 * A number set in tabular figures with proportional separators. Schibsted Grotesk's `tnum` feature also widens
 * the comma and the full stop to a digit's width ("3 , 953"); wrapping them in `.num-sep` (tnum off) keeps the
 * digits on a fixed grid while the separators keep their natural width. The text content is unchanged.
 */
export function Num({ children }: { children: string }) {
  const parts = children.split(/([.,])/);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <span className="num-sep" key={i}>
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </>
  );
}
