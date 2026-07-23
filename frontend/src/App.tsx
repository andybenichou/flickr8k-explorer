import { useEffect, useMemo, useRef, useState } from 'react'
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

/** Read the initial UI state from the URL, so a reload or a shared link lands
 *  exactly where the user left off (which view, query, filter, open image). */
const VIEW_KEYS: View[] = ['grid', 'map', 'stats']
const MODE_KEYS: SearchMode[] = ['semantic', 'text']

function stateFromUrl() {
  const p = new URLSearchParams(window.location.search)
  const view = p.get('view') as View
  const mode = p.get('mode') as SearchMode
  return {
    view: VIEW_KEYS.includes(view) ? view : 'grid',
    query: p.get('q') ?? '',
    mode: MODE_KEYS.includes(mode) ? mode : 'semantic',
    split: p.get('split') ?? '',
    selectedId: p.get('sel'),
  }
}

export default function App() {
  const initial = useRef(stateFromUrl()).current
  const [view, setView] = useState<View>(initial.view)
  const [query, setQuery] = useState(initial.query)
  const [mode, setMode] = useState<SearchMode>(initial.mode)
  const [split, setSplit] = useState(initial.split)
  const [selectedId, setSelectedId] = useState<string | null>(initial.selectedId)

  // Mount a view the first time it is opened, then keep it mounted (hidden) so
  // its scroll position and the map's zoom survive tab switches.
  const [visited, setVisited] = useState<Set<View>>(() => new Set([initial.view]))
  useEffect(() => {
    setVisited((prev) => (prev.has(view) ? prev : new Set(prev).add(view)))
  }, [view])

  // Mirror UI state into the URL (replaceState, so it does not spam history).
  useEffect(() => {
    const p = new URLSearchParams()
    if (view !== 'grid') p.set('view', view)
    if (query) p.set('q', query)
    if (mode !== 'semantic') p.set('mode', mode)
    if (split) p.set('split', split)
    if (selectedId) p.set('sel', selectedId)
    const qs = p.toString()
    window.history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname)
  }, [view, query, mode, split, selectedId])

  // Browser back/forward re-applies whatever state the URL encodes.
  useEffect(() => {
    const onPop = () => {
      const s = stateFromUrl()
      setView(s.view)
      setQuery(s.query)
      setMode(s.mode)
      setSplit(s.split)
      setSelectedId(s.selectedId)
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const dataset = useAsync<DatasetInfo>(() => api.dataset(), [])
  const semanticAvailable = Boolean(dataset.data?.embedding_model)

  // Fall back to text search on a backend without embeddings, so we never fire
  // a semantic query the server cannot answer.
  useEffect(() => {
    if (!semanticAvailable && mode === 'semantic') setMode('text')
  }, [semanticAvailable, mode])

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
          <div className="view-pane" hidden={view !== 'grid'}>
            <ImageGrid
              items={gallery.items}
              selectedId={selectedId}
              loading={gallery.loading}
              error={gallery.error}
              hasMore={gallery.hasMore}
              onLoadMore={gallery.loadMore}
              onSelect={setSelectedId}
            />
          </div>
          {visited.has('map') && (
            <div className="view-pane view-pane--map" hidden={view !== 'map'}>
              <ProjectionMap
                split={split}
                highlightIds={highlightIds}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>
          )}
          {visited.has('stats') && (
            <div className="view-pane" hidden={view !== 'stats'}>
              <StatsPanel />
            </div>
          )}
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
