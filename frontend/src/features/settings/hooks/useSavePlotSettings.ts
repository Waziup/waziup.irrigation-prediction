import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { PlotSettingsInput } from '../../../domain/settings';
import type { PlotContext } from '../../../domain/plot';
import { farmContextKey } from '../../farm-context/selection';
import { saveSettings } from '../settingsActions';
import type { SettingsSnapshot } from '../types';
export function useSavePlotSettings(plotId: string) {
  const client = useQueryClient();
  return useMutation({
    retry: false,
    mutationFn: ({
      baseline,
      input,
    }: {
      baseline: SettingsSnapshot;
      input: PlotSettingsInput;
    }) => saveSettings(plotId, baseline, input),
    onSettled: async () => {
      await client.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === farmContextKey[0] ||
          q.queryKey[2] !== 'settings-editor',
        refetchType: 'none',
      });
    },
    onSuccess: ({ saved }) => {
      // Publish canonical values without refetching/remounting the editor.
      client.setQueryData(['plot', plotId, 'settings-editor'], saved);
      client.setQueryData<PlotContext>(farmContextKey, (context) =>
        context
          ? {
              ...context,
              plots: context.plots.map((plot) =>
                plot.id === saved.plotId
                  ? {
                      ...plot,
                      name: saved.settings.name,
                      areaM2: saved.settings.areaM2,
                      areaUnit: saved.settings.areaUnit,
                    }
                  : plot,
              ),
            }
          : context,
      );
    },
  });
}
