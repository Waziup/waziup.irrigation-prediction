// Optional satellite refinement is described on Growth & Water Demand.
// General advice notices continue to report weather, sensors and model inputs.
export function adviceSources(sources: readonly string[]): string[] {
  return sources.filter(
    (source) => !['satellite', 'eo'].includes(source.toLowerCase()),
  );
}

export function adviceLimitations(limitations: readonly string[]): string[] {
  return limitations.flatMap((limitation) => {
    if (limitation.startsWith('Stale inputs: ')) {
      const sources = adviceSources(
        limitation
          .slice('Stale inputs: '.length)
          .split(',')
          .map((source) => source.trim()),
      );
      return sources.length ? [`Stale inputs: ${sources.join(', ')}`] : [];
    }
    if (
      ['Canopy observation is stale.', 'Insufficient canopy history.'].includes(
        limitation,
      )
    )
      return [];
    return [limitation];
  });
}
