import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Navigate } from 'react-router';
import { farmSetupApi } from './api/farmSetupApi';
import { setupKey } from './types';

export function InstallationGate({ children }: { children: ReactNode }) {
  const setup = useQuery({
    queryKey: setupKey,
    queryFn: ({ signal }) => farmSetupApi.load({ signal }),
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
  if (setup.data?.state === 'empty') return <Navigate to="/setup" replace />;
  // Keep the existing unavailable-page experience during connection errors.
  // Only a confirmed empty registry can redirect to first-time setup.
  return children;
}
