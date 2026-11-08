/** Shared mark and wordmark for navigation and first-time setup. */
export function Brand() {
  return (
    <span className="inline-flex items-center gap-2.5 whitespace-nowrap">
      <img
        src={`${import.meta.env.BASE_URL}wazifarm-logo.svg`}
        alt=""
        className="size-8.5 shrink-0"
      />
      <span>WaziFarm</span>
    </span>
  );
}
