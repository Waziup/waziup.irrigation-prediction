// Presentation contract only. The next slice will load persisted farm/plot context.
export type ShellFarmContext =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'unavailable'; message: string }
  | { status: 'ready'; farmName: string; plotName: string | null };
