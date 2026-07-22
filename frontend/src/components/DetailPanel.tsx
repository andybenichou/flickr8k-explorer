import { api } from '../api'
import { useAsync } from '../hooks'
import type { ImageDetail, SearchResults } from '../types'

interface Props {
  imageId: string
  onSelect: (id: string) => void
  onClose: () => void
  semanticAvailable: boolean
}

function formatBytes(bytes: number): string {
  return bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`
}

export function DetailPanel({ imageId, onSelect, onClose, semanticAvailable }: Props) {
  const detail = useAsync<ImageDetail>(() => api.detail(imageId), [imageId])
  const similar = useAsync<SearchResults | null>(
    () => (semanticAvailable ? api.similar(imageId, 12) : Promise.resolve(null)),
    [imageId, semanticAvailable],
  )

  return (
    <aside className="detail">
      <header className="detail__header">
        <h2>{imageId}</h2>
        <button onClick={onClose} aria-label="Close panel">
          ×
        </button>
      </header>

      {detail.error && <p className="notice notice--error">{detail.error}</p>}
      {detail.data && (
        <>
          <a href={detail.data.image_url} target="_blank" rel="noreferrer" className="detail__image">
            <img src={detail.data.image_url} alt={detail.data.captions[0]?.text ?? imageId} />
          </a>

          <dl className="detail__meta">
            <div>
              <dt>Split</dt>
              <dd>{detail.data.split}</dd>
            </div>
            <div>
              <dt>Resolution</dt>
              <dd>
                {detail.data.width} × {detail.data.height}
              </dd>
            </div>
            <div>
              <dt>Aspect</dt>
              <dd>{detail.data.aspect_ratio}</dd>
            </div>
            <div>
              <dt>File size</dt>
              <dd>{formatBytes(detail.data.file_size)}</dd>
            </div>
            {detail.data.cluster !== null && (
              <div>
                <dt>Cluster</dt>
                <dd>#{detail.data.cluster}</dd>
              </div>
            )}
          </dl>

          <h3>Captions ({detail.data.captions.length})</h3>
          <ol className="detail__captions">
            {detail.data.captions.map((caption) => (
              <li key={caption.index}>
                {caption.text}
                <span className="detail__words">{caption.n_words}w</span>
              </li>
            ))}
          </ol>
        </>
      )}

      {semanticAvailable && (
        <>
          <h3>Nearest neighbours</h3>
          <p className="detail__hint">
            Closest images in CLIP space. Near-identical neighbours usually mean duplicates or
            over-represented scenes.
          </p>
          <div className="detail__similar">
            {similar.data?.items.map((item) => (
              <button key={item.id} onClick={() => onSelect(item.id)} title={item.caption}>
                <img src={item.thumb_url} alt={item.caption} loading="lazy" />
                <span>{item.score?.toFixed(3)}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </aside>
  )
}
