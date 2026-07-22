// Thin typed wrapper over the backend. One place to change if routes move.

import type {
  DatasetInfo,
  DatasetStats,
  ImageDetail,
  ImagePage,
  Projection,
  SearchMode,
  SearchResults,
} from './types'

async function get<T>(path: string, params: Record<string, string | number | undefined> = {}): Promise<T> {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') query.set(key, String(value))
  }
  const url = query.toString() ? `${path}?${query}` : path
  const response = await fetch(url)
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  dataset: () => get<DatasetInfo>('/api/dataset'),

  images: (params: { split?: string; offset: number; limit: number }) =>
    get<ImagePage>('/api/images', params),

  detail: (id: string) => get<ImageDetail>(`/api/images/${id}`),

  similar: (id: string, limit = 12) => get<SearchResults>(`/api/images/${id}/similar`, { limit }),

  search: (params: { q: string; mode: SearchMode; split?: string; limit?: number }) =>
    get<SearchResults>('/api/search', { limit: 48, ...params }),

  projection: (split?: string) => get<Projection>('/api/projection', { split }),

  stats: () => get<DatasetStats>('/api/stats'),
}
