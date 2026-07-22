import { useMemo, useState } from 'react'
import { api } from './api'
import { DetailPanel } from './components/DetailPanel'
import { ImageGrid } from './components/ImageGrid'
import { ProjectionMap } from './components/ProjectionMap'
import { SearchBar } from './components/SearchBar'
import { StatsPanel } from './components/StatsPanel'
import { useAsync, useDebounced, useGallery } from './hooks'
import type { DatasetInfo, SearchMode, View } from './types'

const VIEWS: { key: View; label: string; hint: string }[] = [
  { key: 'grid', label: 'Browse', hint: 'Grid of samples, searchable' },
  { key: 'map', label: 'Map', hint: '2D projection of the embedding space' },
  { key: 'stats', label: 'Stats', hint: 'Dataset composition' },
]

export default function App() {
  const [view, setView] = useState<View>('grid')
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('semantic')
  const [split, setSplit] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const dataset = useAsync<DatasetInfo>(() => api.dataset(), [])
  const semanticAvailable = Boolean(dataset.data?.embedding_model)

  // Debounced so typing does not fire a CLIP query per keystroke.
  const debouncedQuery = useDebounced(query)
  const gallery = useGallery(debouncedQuery, mode, split || undefined)

  // The map dims everything that is not a search hit, which turns a text query
  // into a "where does this concept live in the dataset" question.
  const highlightIds = useMemo(
    () => (gallery.ranked ? new Set(gallery.items.map((item) => item.id)) : new Set<string>()),
    [gallery.ranked, gallery.items],
  )

  if (dataset.error) {
    return (
      <div className="boot-error">
        <h1>Backend unavailable</h1>
        <p>{dataset.error}</p>
        <p>
          Make sure the dataset has been ingested (<code>uv run python -m backend.ingest</code>) and
          the API is running (<code>uv run uvicorn backend.app.api:app</code>).
        </p>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <h1>Flickr8k Explorer</h1>
          {dataset.data && (
            <span>
              {dataset.data.n_images.toLocaleString()} images · {dataset.data.n_captions.toLocaleString()} captions
              {dataset.data.embedding_model && ` · ${dataset.data.embedding_dim}-d CLIP`}
            </span>
          )}
        </div>
        <nav className="segmented">
          {VIEWS.map((item) => (
            <button
              key={item.key}
              className={view === item.key ? 'active' : ''}
              onClick={() => setView(item.key)}
              title={item.hint}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      {view !== 'stats' && (
        <SearchBar
          query={query}
          onQueryChange={setQuery}
          mode={mode}
          onModeChange={setMode}
          split={split}
          splits={dataset.data?.splits ?? []}
          onSplitChange={setSplit}
          semanticAvailable={semanticAvailable}
          resultCount={gallery.ranked ? gallery.items.length : gallery.total}
          ranked={gallery.ranked}
        />
      )}

      <div className="workspace">
        <main className="workspace__main">
          {view === 'grid' && (
            <ImageGrid
              items={gallery.items}
              selectedId={selectedId}
              loading={gallery.loading}
              error={gallery.error}
              hasMore={gallery.hasMore}
              onLoadMore={gallery.loadMore}
              onSelect={setSelectedId}
            />
          )}
          {view === 'map' && (
            <ProjectionMap
              split={split}
              highlightIds={highlightIds}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
          {view === 'stats' && <StatsPanel />}
        </main>

        {selectedId && (
          <DetailPanel
            imageId={selectedId}
            onSelect={setSelectedId}
            onClose={() => setSelectedId(null)}
            semanticAvailable={semanticAvailable}
          />
        )}
      </div>
    </div>
  )
}
