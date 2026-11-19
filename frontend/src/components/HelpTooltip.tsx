import { useEffect, useId, useRef, useState } from 'react';
import type { CSSProperties } from 'react';

export function HelpTooltip({
  label,
  text,
  id,
}: {
  label: string;
  text: string;
  id?: string;
}) {
  const generatedId = useId();
  const descriptionId = id ?? generatedId;
  const button = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);
  function handleToggle(isOpen: boolean) {
    setOpen(isOpen);
    if (!isOpen || !button.current || !panel.current) {
      setPosition(null);
      return;
    }
    // Measure while invisible so there is no flash at the browser's default
    // popover position. Native top-layer rendering avoids ancestor clipping.
    const anchor = button.current.getBoundingClientRect();
    const popup = panel.current.getBoundingClientRect();
    const padding = 16;
    const gap = 8;
    const left = Math.max(
      padding,
      Math.min(
        anchor.right - popup.width,
        window.innerWidth - popup.width - padding,
      ),
    );
    const below = anchor.bottom + gap;
    const top =
      below + popup.height <= window.innerHeight - padding
        ? below
        : Math.max(padding, anchor.top - popup.height - gap);
    setPosition({ left, top });
  }
  useEffect(() => {
    if (!open) return;
    function dismissOutside(event: Event) {
      if (
        event.target instanceof Node &&
        !button.current?.contains(event.target) &&
        !panel.current?.contains(event.target)
      )
        panel.current?.hidePopover();
    }
    const dismiss = () => panel.current?.hidePopover();
    document.addEventListener('focusin', dismissOutside);
    window.addEventListener('resize', dismiss);
    window.addEventListener('scroll', dismiss, true);
    return () => {
      document.removeEventListener('focusin', dismissOutside);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('scroll', dismiss, true);
    };
  }, [open]);
  return (
    <span className="relative inline-flex shrink-0 items-center align-middle">
      <button
        ref={button}
        type="button"
        popoverTarget={`${descriptionId}-popup`}
        aria-label={`Help for ${label}`}
        aria-describedby={descriptionId}
        aria-expanded={open}
        className="flex size-6 cursor-pointer items-center justify-center rounded-full border border-[#a8b4c4] bg-white text-xs font-semibold tracking-normal text-[#52627a] normal-case hover:bg-[#f3f7f2] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#315f48]"
        onClick={(event) => event.stopPropagation()}
      >
        ?
      </button>
      <span id={descriptionId} className="sr-only">
        {text}
      </span>
      <div
        ref={panel}
        id={`${descriptionId}-popup`}
        popover="auto"
        role="tooltip"
        onToggle={(event) => handleToggle(event.newState === 'open')}
        className={`fixed inset-auto top-[var(--help-top)] left-[var(--help-left)] z-[100] m-0 max-h-[calc(100dvh-2rem)] w-72 max-w-[calc(100vw-2rem)] overflow-y-auto rounded-lg border border-[#d8dee8] bg-white p-3 text-left text-xs font-normal leading-relaxed tracking-normal text-[#374151] normal-case shadow-lg ${open && position ? 'visible' : 'invisible'}`}
        // Runtime coordinates are measured pixels; all visual styling uses Tailwind.
        // eslint-disable-next-line no-restricted-syntax
        style={
          {
            '--help-left': `${position?.left ?? 0}px`,
            '--help-top': `${position?.top ?? 0}px`,
          } as CSSProperties
        }
      >
        {text}
      </div>
    </span>
  );
}
