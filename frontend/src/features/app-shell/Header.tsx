import { Link } from 'react-router';
import { Icon } from '../../components/Icon';
export function Header({
  title,
  onOpenNavigation,
}: {
  title: string;
  onOpenNavigation?: () => void;
}) {
  return (
    <header className="relative flex min-h-15.5 items-center gap-3 border-b border-[#e7ebe0] bg-[#fcfdf9] px-4 py-3 md:px-8 xl:px-10">
      <button
        type="button"
        aria-label="Open navigation"
        aria-haspopup="dialog"
        onClick={onOpenNavigation}
        className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-gray-300 text-[#174f2a] md:hidden"
      >
        <Icon name="menu" />
      </button>
      <h1
        className="min-w-0 flex-1 text-sm font-medium text-[#73836b]"
        id="page-title"
      >
        {title}
      </h1>
      <Link
        to="/settings"
        aria-label="Farm settings"
        className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-[#e5eadd] bg-white text-[#6c7e61] hover:bg-[#e7f2e4]"
      >
        <Icon name="settings" />
      </Link>
    </header>
  );
}
