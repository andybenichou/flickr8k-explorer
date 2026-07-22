import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { Projection, ProjectionPoint } from '../types'

interface Props {
  split: string
  highlightIds: Set<string>
  selectedId: string | null
  onSelect: (id: string) => void
}

// Categorical palette, readable on the dark background and distinguishable
// without relying on hue alone at small point sizes.
const CLUSTER_COLORS = [
  '#7dd3fc', '#fca5a5', '#86efac', '#fcd34d', '#c4b5fd', '#f9a8d4',
  '#5eead4', '#fdba74', '#a3e635', '#93c5fd', '#f0abfc', '#d6d3d1',
]

const PADDING = 24

export function ProjectionMap({ split, highlightIds, selectedId, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  const [hovered, setHovered] = useState<{ point: ProjectionPoint; x: number; y: number } | null>(null)

  const projection = useAsync<Projection>(() => api.projection(split || undefined), [split])
  const points = useMemo(() => projection.data?.points ?? [], [projection.data])

  // Track the container size so the map fills the available space.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas?.parentElement) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(canvas.parentElement)
    return () => observer.disconnect()
  }, [])

  const toPixel = useMemo(() => {
    const w = size.width - PADDING * 2
    const h = size.height - PADDING * 2
    return (point: ProjectionPoint) => ({
      x: PADDING + point.x * w,
      y: PADDING + (1 - point.y) * h,
    })
  }, [size])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ratio = window.devicePixelRatio || 1
    canvas.width = size.width * ratio
    canvas.height = size.height * ratio
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
    ctx.clearRect(0, 0, size.width, size.height)

    // Two passes so highlighted points always land on top of the dimmed cloud.
    const dimming = highlightIds.size > 0
    const passes = dimming ? [false, true] : [true]
    for (const highlightPass of passes) {
      for (const point of points) {
        const isHit = !dimming || highlightIds.has(point.id)
        if (isHit !== highlightPass) continue
        const { x, y } = toPixel(point)
        ctx.globalAlpha = isHit ? 0.85 : 0.5
        ctx.fillStyle = isHit
          ? CLUSTER_COLORS[(point.cluster ?? 0) % CLUSTER_COLORS.length]
          : '#3a4150'
        ctx.beginPath()
        ctx.arc(x, y, isHit && dimming ? 4 : 2.2, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Draw the selected point last so it is never hidden behind the cloud.
    const selected = points.find((p) => p.id === selectedId)
    if (selected) {
      const { x, y } = toPixel(selected)
      ctx.globalAlpha = 1
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.globalAlpha = 1
  }, [points, size, toPixel, highlightIds, selectedId])

  /** Nearest point to the cursor, within a small radius. 8k points is a cheap scan. */
  function nearest(clientX: number, clientY: number) {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const mx = clientX - rect.left
    const my = clientY - rect.top
    let best: ProjectionPoint | null = null
    let bestDistance = 12 * 12
    for (const point of points) {
      const { x, y } = toPixel(point)
      const distance = (x - mx) ** 2 + (y - my) ** 2
      if (distance < bestDistance) {
        bestDistance = distance
        best = point
      }
    }
    return best ? { point: best, x: mx, y: my } : null
  }

  if (projection.loading) return <p className="notice">Loading projection…</p>
  if (projection.error) return <p className="notice notice--error">{projection.error}</p>
  if (points.length === 0)
    return (
      <p className="notice">
        No 2D projection stored. Re-run the ingestion without <code>--skip-projection</code>.
      </p>
    )

  return (
    <div className="map">
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        onMouseMove={(event) => setHovered(nearest(event.clientX, event.clientY))}
        onMouseLeave={() => setHovered(null)}
        onClick={(event) => {
          const hit = nearest(event.clientX, event.clientY)
          if (hit) onSelect(hit.point.id)
        }}
      />
      {hovered && (
        <div
          className="map__tooltip"
          style={{
            left: Math.min(hovered.x + 14, size.width - 190),
            top: Math.min(hovered.y + 14, size.height - 190),
          }}
        >
          <img src={`/media/thumbs/${hovered.point.id}.jpg`} alt={hovered.point.id} />
          <span>
            {hovered.point.id} · cluster #{hovered.point.cluster}
          </span>
        </div>
      )}
      <div className="map__legend">
        <strong>UMAP of CLIP image embeddings</strong>
        <span>
          {points.length.toLocaleString()} images · colour = KMeans cluster
          {highlightIds.size > 0 && ` · ${highlightIds.size} search hits highlighted`}
        </span>
      </div>
    </div>
  )
}
