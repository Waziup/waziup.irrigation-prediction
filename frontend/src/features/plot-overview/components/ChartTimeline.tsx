import { useEffect, useRef, useState } from 'react';
import { panRange, type TimeRange } from '../chart/navigation';

export function ChartTimeline({
  domain,
  range,
  onChange,
}: {
  domain: TimeRange;
  range: TimeRange;
  onChange: (range: TimeRange) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1);
  const span = domain.end - domain.start;
  const visible = range.end - range.start;
  const contentWidth = Math.max(
    width,
    Math.round((width * span) / Math.max(1, visible)),
  );
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const element = viewport.current;
    if (!element || span <= visible) return;
    const left =
      ((range.start - domain.start) / (span - visible)) *
      (element.scrollWidth - element.clientWidth);
    if (Math.abs(element.scrollLeft - left) > 1) element.scrollLeft = left;
  }, [domain.start, range.start, span, visible, contentWidth]);
  return (
    <div
      ref={viewport}
      role="region"
      aria-label="Prediction chart timeline"
      tabIndex={0}
      aria-keyshortcuts="ArrowLeft ArrowRight Home End"
      onKeyDown={(event) => {
        if (event.key === 'Home')
          onChange({ start: domain.start, end: domain.start + visible });
        else if (event.key === 'End')
          onChange({ start: domain.end - visible, end: domain.end });
        else if (event.key === 'ArrowLeft')
          onChange(panRange(range, domain, -0.1));
        else if (event.key === 'ArrowRight')
          onChange(panRange(range, domain, 0.1));
        else return;
        event.preventDefault();
      }}
      className="mt-2 h-5 min-w-0 overflow-x-scroll overscroll-x-contain [scrollbar-width:thin] focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-[#74936c]"
      onScroll={(event) => {
        const element = event.currentTarget;
        const travel = element.scrollWidth - element.clientWidth;
        if (travel <= 0 || span <= visible) return;
        const start =
          domain.start + (element.scrollLeft / travel) * (span - visible);
        // Native scrollLeft is pixel-rounded. Do not feed programmatic sync
        // back into the chart when it already represents this same position.
        const tolerance = (span - visible) / travel + 1;
        if (Math.abs(start - range.start) <= tolerance) return;
        onChange({ start, end: start + visible });
      }}
    >
      <svg
        width={contentWidth}
        height={1}
        aria-hidden="true"
        className="block max-w-none shrink-0"
      />
    </div>
  );
}
