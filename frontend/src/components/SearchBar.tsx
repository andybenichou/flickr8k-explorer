import type { SearchMode } from '../types'

interface Props {
  query: string
  onQueryChange: (value: string) => void
  mode: SearchMode
  onModeChange: (mode: SearchMode) => void
  split: string
  splits: string[]
  onSplitChange: (split: string) => void
  semanticAvailable: boolean
  resultCount: number
  ranked: boolean
}

const PLACEHOLDERS: Record<SearchMode, string> = {
  semantic: 'Describe a scene: "a dog jumping over a fence at the beach"',
  text: 'Match caption words: dog beach',
}

export function SearchBar({
  query,
  onQueryChange,
  mode,
  onModeChange,
  split,
  splits,
  onSplitChange,
  semanticAvailable,
  resultCount,
  ranked,
}: Props) {
  return (
    <div className="searchbar">
      <div className="searchbar__input">
        {/* `text`, not `search`: the native clear button would duplicate ours. */}
        <input
          type="text"
          value={query}
          placeholder={PLACEHOLDERS[mode]}
          onChange={(event) => onQueryChange(event.target.value)}
          aria-label="Search the dataset"
        />
        {query && (
          <button className="searchbar__clear" onClick={() => onQueryChange('')} aria-label="Clear search">
            ×
          </button>
        )}
      </div>

      <div className="segmented" role="group" aria-label="Search mode">
        <button
          className={mode === 'semantic' ? 'active' : ''}
          onClick={() => onModeChange('semantic')}
          disabled={!semanticAvailable}
          aria-pressed={mode === 'semantic'}
          title={
            semanticAvailable
              ? 'CLIP embedding similarity: finds images by meaning, not words'
              : 'Run the ingestion with embeddings enabled to use semantic search'
          }
        >
          Semantic
        </button>
        <button
          className={mode === 'text' ? 'active' : ''}
          onClick={() => onModeChange('text')}
          aria-pressed={mode === 'text'}
          title="BM25 full-text match over the five captions"
        >
          Caption text
        </button>
      </div>

      <select value={split} onChange={(event) => onSplitChange(event.target.value)} aria-label="Split">
        <option value="">All splits</option>
        {splits.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>

      <span className="searchbar__count">
        {ranked ? `${resultCount} ranked` : `${resultCount.toLocaleString()} images`}
      </span>
    </div>
  )
}
