/** Shapes reserve the default hero and memory card's footprint while the first worker result arrives. */
export function ResultSkeleton({ roofline }: { roofline: boolean }) {
  return (
    <div className="result-skeleton" data-testid="result-skeleton" aria-hidden="true">
      <div className={`skeleton-grid ${roofline ? '' : 'no-roofline'}`}>
        <div className="skeleton-answers">
          {[0, 1].map((phase) => (
            <div className="skeleton-answer" key={phase}>
              <span className="skeleton-shape skeleton-label" />
              <span className="skeleton-shape skeleton-number" />
              <span className="skeleton-shape skeleton-label" />
            </div>
          ))}
          <span className="skeleton-shape skeleton-chip" />
        </div>
        {roofline && <div className="skeleton-chart skeleton-shape" />}
      </div>
      <div className="skeleton-theory">
        <span className="skeleton-shape" />
      </div>
      <div className="skeleton-memory">
        <span className="skeleton-shape skeleton-label" />
        <span className="skeleton-shape skeleton-bar" />
        <span className="skeleton-shape skeleton-bar" />
        <div className="skeleton-legend">
          {[0, 1, 2].map((item) => (
            <span key={item} className="skeleton-shape" />
          ))}
        </div>
      </div>
    </div>
  );
}
