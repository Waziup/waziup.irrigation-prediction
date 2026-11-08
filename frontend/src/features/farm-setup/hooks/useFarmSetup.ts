import { useMutation } from '@tanstack/react-query';
import { useRef, useState } from 'react';
import { farmSetupApi } from '../api/farmSetupApi';
import { matchesSavedDraft, saveSetup, SetupSaveError } from '../setupActions';
import type { Installation, SetupDraft } from '../types';

export function useFarmSetup(initial: Installation, mode: 'farm' | 'plot') {
  const [baseline, setBaseline] = useState(initial);
  const [mustReload, setMustReload] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [reconciled, setReconciled] = useState<Installation | null>(null);
  const [reloadError, setReloadError] = useState('');
  const lock = useRef(false);
  const mutation = useMutation({
    mutationFn: (draft: SetupDraft) => saveSetup(draft, baseline, mode),
    retry: false,
  });
  return {
    baseline,
    clearError: () => {
      mutation.reset();
      setReloadError('');
    },
    mustReload,
    reloading,
    reconciled,
    reloadError,
    error: mutation.error,
    busy: mutation.isPending || reloading,
    async save(draft: SetupDraft) {
      if (lock.current || mustReload || reloading) return null;
      lock.current = true;
      try {
        return await mutation.mutateAsync(draft);
      } catch (error) {
        setMustReload(error instanceof SetupSaveError && error.requiresReload);
        throw error;
      } finally {
        lock.current = false;
      }
    },
    async reload(draft: SetupDraft) {
      if (lock.current) return;
      lock.current = true;
      setReloading(true);
      setReloadError('');
      try {
        const saved = await farmSetupApi.load();
        setBaseline(saved);
        if (
          (mode === 'farm' && saved.state === 'complete') ||
          matchesSavedDraft(draft, saved, mode)
        ) {
          setReconciled(saved);
          setMustReload(true);
        } else {
          setMustReload(false);
          mutation.reset();
        }
      } catch (cause) {
        setReloadError(
          cause instanceof Error
            ? cause.message
            : 'Saved setup could not be loaded.',
        );
      } finally {
        lock.current = false;
        setReloading(false);
      }
    },
  };
}
