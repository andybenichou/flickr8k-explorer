import { api } from '../api'
import { useAsync } from '../hooks'
import type { Bucket, DatasetStats } from '../types'

function BarChart({ title, hint, buckets }: { title: string; hint?: string; buckets: Bucket[] }) {
  const max = Math.max(1, ...buckets.map((b) => b.count))
  return (
    <section className="chart">
      <h3>{title}</h3>
      {hint && <p className="chart__hint">{hint}</p>}
      <ul>
        {buckets.map((bucket) => (
          <li key={bucket.label}>
            <span className="chart__label">{bucket.label}</span>
            <span className="chart__bar">
              <span style={{ width: `${(bucket.count / max) * 100}%` }} />
            </span>
            <span className="chart__value">{bucket.count.toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function StatsPanel() {
  const stats = useAsync<DatasetStats>(() => api.stats(), [])

  if (stats.loading) return <p className="notice">Computing statistics…</p>
  if (stats.error) return <p className="notice notice--error">{stats.error}</p>
  if (!stats.data) return null

  const s = stats.data
  return (
    <div className="stats">
      <div className="stats__kpis">
        <div>
          <strong>{s.n_images.toLocaleString()}</strong>
          <span>images</span>
        </div>
        <div>
          <strong>{s.n_captions.toLocaleString()}</strong>
          <span>captions</span>
        </div>
        <div>
          <strong>{s.mean_caption_words}</strong>
          <span>words / caption</span>
        </div>
        <div>
          <strong>{s.vocabulary_size.toLocaleString()}</strong>
          <span>vocabulary</span>
        </div>
      </div>

      <div className="stats__charts">
        <BarChart title="Images per split" buckets={s.images_per_split} />
        <BarChart
          title="Caption length"
          hint="Words per caption. A long tail here means annotation style varies across the set."
          buckets={s.caption_length_histogram}
        />
        <BarChart
          title="Most common resolutions"
          hint="Flickr8k is not resolution-normalised; check this before assuming a fixed input size."
          buckets={s.resolution_histogram}
        />
        <BarChart
          title="Most frequent caption words"
          hint="Stopwords removed. The head of this distribution is the dataset's content bias."
          buckets={s.top_words}
        />
      </div>
    </div>
  )
}
