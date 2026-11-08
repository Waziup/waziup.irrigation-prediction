import { Link } from 'react-router';
export function NotFoundPage() {
  return (
    <section className="min-w-0 rounded-xl border border-gray-200 bg-white p-5 shadow-xs sm:p-8 [&>h2]:mt-0">
      <h2 className="mt-4 mb-2.5 text-xl font-semibold">Page not found</h2>
      <p className="my-3 leading-relaxed text-gray-600">
        The requested page is unavailable.
      </p>
      <Link
        className="text-[#2d5a27] underline-offset-4 hover:underline"
        to="/dashboard"
      >
        Return to farm overview
      </Link>
    </section>
  );
}
