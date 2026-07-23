// Mirrors the pydantic schemas in backend/app/models.py.

export interface ImageSummary {
  id: string
  split: string
  width: number
  height: number
  thumb_url: string
  caption: string
  score: number | null
}

export interface Caption {
  index: number
  text: string
  n_words: number
}

export interface ImageDetail {
  id: string
  split: string
  width: number
  height: number
  file_size: number
  aspect_ratio: number
  image_url: string
  thumb_url: string
  captions: Caption[]
  umap: [number, number] | null
  cluster: number | null
}

export interface ImagePage {
  items: ImageSummary[]
  total: number
  offset: number
  limit: number
}

export interface SearchResults {
  items: ImageSummary[]
  total: number
  mode: string
  query: string
}

export interface ProjectionPoint {
  id: string
  x: number
  y: number
  cluster: number | null
  split: string
}

export interface ClusterInfo {
  id: number
  label: string
  size: number
}

export interface Projection {
  points: ProjectionPoint[]
  n_clusters: number
  clusters: ClusterInfo[]
}

export interface Bucket {
  label: string
  count: number
}

export interface DatasetStats {
  n_images: number
  n_captions: number
  images_per_split: Bucket[]
  caption_length_histogram: Bucket[]
  resolution_histogram: Bucket[]
  top_words: Bucket[]
  mean_caption_words: number
  vocabulary_size: number
}

export interface DatasetInfo {
  name: string
  n_images: number
  n_captions: number
  splits: string[]
  embedding_model: string | null
  embedding_dim: number | null
  has_projection: boolean
  ingested_at: string | null
}

export type SearchMode = 'semantic' | 'text'
export type View = 'grid' | 'map' | 'stats'
