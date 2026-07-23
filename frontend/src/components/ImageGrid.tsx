import { useOnVisible } from '../hooks'
import type { ImageSummary } from '../types'
import { Spinner } from './Spinner'

interface CardProps {
  image: ImageSummary
  selected: boolean
  onSelect: (id: string) => void
}

function ImageCard({ image, selected, onSelect }: CardProps) {
  return (
    <button
      className={`card${selected ? ' card--selected' : ''}`}
      onClick={() => onSelect(image.id)}
      title={image.caption}
    >
      <img src={image.thumb_url} alt={image.caption} loading="lazy" decoding="async" />
      <span className="card__caption">{image.caption}</span>
      <span className="card__meta">
        <span className={`tag tag--${image.split}`}>{image.split}</span>
        {image.score !== null && <span className="tag tag--score">{image.score.toFixed(3)}</span>}
      </span>
    </button>
  )
}

interface GridProps {
  items: ImageSummary[]
  selectedId: string | null
  loading: boolean
  error: string | null
  hasMore: boolean
  onLoadMore: () => void
  onSelect: (id: string) => void
  emptyMessage?: string
}

export function ImageGrid({
  items,
  selectedId,
  loading,
  error,
  hasMore,
  onLoadMore,
  onSelect,
  emptyMessage = 'No images match this query.',
}: GridProps) {
  // Infinite scroll rather than windowing: the browser already lazy-loads
  // offscreen thumbnails, which keeps the DOM cheap without a virtualisation
  // library and its responsive-column complexity.
  const sentinel = useOnVisible(onLoadMore, hasMore && !loading)

  if (error) return <p className="notice notice--error">{error}</p>
  if (!loading && items.length === 0) return <p className="notice">{emptyMessage}</p>
  // First page still in flight: a centred spinner rather than an empty pane.
  if (loading && items.length === 0) return <Spinner block />

  return (
    <>
      <div className="grid">
        {items.map((image) => (
          <ImageCard
            key={image.id}
            image={image}
            selected={image.id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </div>
      {loading && <Spinner label="Loading more…" />}
      <div ref={sentinel} className="sentinel" />
    </>
  )
}
