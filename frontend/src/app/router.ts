import { createHashRouter, Navigate, Outlet } from 'react-router';
import { createElement } from 'react';
import { NotFoundPage } from './App';
import { SettingsPage } from '../features/settings/SettingsPage';
import { PlotOverviewPage } from '../features/plot-overview/PlotOverviewPage';
import { GrowthOverviewPage } from '../features/plot-overview/GrowthOverviewPage';
import { FarmOverviewPage } from '../features/farm-overview/FarmOverviewPage';
import { FarmContextLayout } from '../features/farm-context/FarmContextLayout';
import {
  PageErrorBoundary,
  ShellErrorBoundary,
} from '../features/app-shell/ErrorBoundary';

export const router = createHashRouter([
  {
    path: '/setup',
    lazy: async () => {
      const page = await import('../features/farm-setup/FarmSetupPage');
      return { Component: page.FarmSetupPage };
    },
    ErrorBoundary: PageErrorBoundary,
  },
  {
    path: '/setup/plot',
    lazy: async () => {
      const page = await import('../features/farm-setup/FarmSetupPage');
      return { Component: page.PlotSetupPage };
    },
    ErrorBoundary: PageErrorBoundary,
  },
  {
    path: '/settings/farm/new',
    lazy: async () => {
      const page = await import('../features/farm-setup/FarmSetupPage');
      return { Component: page.FarmSetupPage };
    },
    ErrorBoundary: PageErrorBoundary,
  },
  {
    path: '/settings/plots/new',
    lazy: async () => {
      const page = await import('../features/farm-setup/FarmSetupPage');
      return { Component: page.PlotSetupPage };
    },
    ErrorBoundary: PageErrorBoundary,
  },
  {
    path: '/',
    Component: FarmContextLayout,
    ErrorBoundary: ShellErrorBoundary,
    children: [
      {
        Component: Outlet,
        ErrorBoundary: PageErrorBoundary,
        children: [
          {
            index: true,
            element: createElement(Navigate, {
              to: '/dashboard',
              replace: true,
            }),
          },
          { path: 'dashboard', Component: FarmOverviewPage },
          {
            path: 'map',
            element: createElement(Navigate, {
              to: '/dashboard',
              replace: true,
            }),
          },
          {
            path: 'settings/boundaries',
            lazy: async () => {
              const page =
                await import('../features/settings/BoundarySettingsPage');
              return { Component: page.BoundarySettingsPage };
            },
          },
          { path: 'plot', Component: PlotOverviewPage },
          { path: 'phenology', Component: GrowthOverviewPage },
          { path: 'settings', Component: SettingsPage },
          { path: '*', Component: NotFoundPage },
        ],
      },
    ],
  },
]);
