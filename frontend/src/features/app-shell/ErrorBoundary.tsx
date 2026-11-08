import { Link, useRouteError } from 'react-router';
import { Icon } from '../../components/Icon';
import { ApiError } from '../../api/errors';

export function PageErrorBoundary() {
  const error = useRouteError();
  const message =
    error instanceof ApiError
      ? error.message
      : 'This page could not be loaded. Please try again.';
  return (
    <section
      className="min-w-0 rounded-xl border border-gray-200 bg-white p-5 shadow-xs sm:p-8 [&>h2]:mt-0"
      aria-labelledby="page-error-title"
    >
      <Icon name="alert" />
      <h2 className="mt-4 mb-2.5 text-xl font-semibold" id="page-error-title">
        Unable to load this page
      </h2>
      <p className="my-3 leading-relaxed text-gray-600" role="alert">
        {message}
      </p>
      <div className="flex flex-wrap items-center gap-5">
        <Link
          className="text-[#2d5a27] underline-offset-4 hover:underline"
          to="/dashboard"
        >
          Return to farm overview
        </Link>
        <button
          className="min-h-11 cursor-pointer rounded-lg border border-gray-200 bg-white px-4 py-3 text-[#2d5a27] hover:bg-gray-50 disabled:cursor-default disabled:opacity-55"
          type="button"
          onClick={() => window.location.reload()}
        >
          Reload application
        </button>
      </div>
    </section>
  );
}
// Last-resort route boundary: still works if the shared shell itself fails.
export function ShellErrorBoundary() {
  return (
    <main className="mx-auto my-[10vh] max-w-160 p-6">
      <p className="flex items-center gap-2.5 text-2xl font-extrabold text-[#2d5a27] no-underline">
        WaziFarm
      </p>
      <h1 className="text-2xl font-semibold leading-tight">
        Unable to open the application
      </h1>
      <p className="my-3 leading-relaxed text-gray-600" role="alert">
        Please reload the application and try again.
      </p>
      <button
        className="min-h-11 cursor-pointer rounded-lg border border-gray-200 bg-white px-4 py-3 text-[#2d5a27] hover:bg-gray-50 disabled:cursor-default disabled:opacity-55"
        type="button"
        onClick={() => window.location.reload()}
      >
        Reload application
      </button>
    </main>
  );
}
