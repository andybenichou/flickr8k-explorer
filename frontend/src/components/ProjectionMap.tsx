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

const colorFor = (cluster: number | null) =>
  CLUSTER_COLORS[(cluster ?? 0) % CLUSTER_COLORS.length]

const PADDING = 24

// Tooltip dimensions used to clamp it inside the map. TIP_W is the CSS width;
// TIP_H is a conservative upper bound on its variable height.
const TIP_W = 176
const TIP_H = 190

export function ProjectionMap({ split, highlightIds, selectedId, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  const [hovered, setHovered] = useState<{ point: ProjectionPoint; x: number; y: number } | null>(null)
  // Set by hovering a legend row: isolates one cluster so its shape stands out.
  const [activeCluster, setActiveCluster] = useState<number | null>(null)
  // Pan/zoom viewport: screen = base * k + (x, y). Dense blobs overlap at 1x,
  // so zooming in is the only way to reach an individual point.
  const [view, setView] = useState({ k: 1, x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  // Boolean mirror of `drag` used only for the cursor: a ref mutation does not
  // re-render, so without this the 'grabbing' cursor lags one frame.
  const [dragging, setDragging] = useState(false)
  // Touch gesture state: one finger pans, two fingers pinch-zoom. `distance` is
  // the previous finger spread, used to derive the pinch scale ratio per move.
  const touch = useRef<{ mode: 'pan' | 'pinch'; x: number; y: number; moved: boolean; distance: number } | null>(null)
  // Timestamp of the last one-finger tap, so a quick second tap resets the view.
  const lastTap = useRef(0)
  // Collapsible overlay panel: on a phone it otherwise covers most of the map.
  const [panelOpen, setPanelOpen] = useState(true)

  const projection = useAsync<Projection>(() => api.projection(split || undefined), [split])
  const points = useMemo(() => projection.data?.points ?? [], [projection.data])
  const clusters = useMemo(() => projection.data?.clusters ?? [], [projection.data])
  const labelFor = useMemo(() => {
    const map = new Map(clusters.map((c) => [c.id, c.label]))
    return (cluster: number | null) => (cluster == null ? '—' : map.get(cluster) ?? `cluster ${cluster}`)
  }, [clusters])

  // A point is "in focus" if it matches the current search hits, or the cluster
  // the user is hovering in the legend. With neither active, nothing is dimmed.
  const searching = highlightIds.size > 0
  const focus = useMemo(() => {
    if (searching) return (p: ProjectionPoint) => highlightIds.has(p.id)
    if (activeCluster != null) return (p: ProjectionPoint) => p.cluster === activeCluster
    return null
  }, [searching, highlightIds, activeCluster])

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

  // Data coordinates -> base canvas pixels (before pan/zoom).
  const toBase = useMemo(() => {
    const w = size.width - PADDING * 2
    const h = size.height - PADDING * 2
    return (point: ProjectionPoint) => ({
      x: PADDING + point.x * w,
      y: PADDING + (1 - point.y) * h,
    })
  }, [size])

  // Base pixels -> on-screen pixels, applying the current viewport.
  const toPixel = useMemo(() => {
    return (point: ProjectionPoint) => {
      const b = toBase(point)
      return { x: b.x * view.k + view.x, y: b.y * view.k + view.y }
    }
  }, [toBase, view])

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

    // Two passes so focused points always land on top of the dimmed cloud.
    const passes = focus ? [false, true] : [true]
    for (const focusPass of passes) {
      for (const point of points) {
        const isHit = !focus || focus(point)
        if (isHit !== focusPass) continue
        const { x, y } = toPixel(point)
        ctx.globalAlpha = isHit ? 0.85 : 0.45
        ctx.fillStyle = isHit ? colorFor(point.cluster) : '#3a4150'
        ctx.beginPath()
        const radius = (isHit && focus ? 3.6 : 2.2) * Math.min(1.6, Math.sqrt(view.k))
        ctx.arc(x, y, radius, 0, Math.PI * 2)
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
  }, [points, size, toPixel, focus, selectedId, view])

  // Wheel zoom, centred on the cursor so the point under it stays put. Attached
  // manually because React's onWheel is passive and cannot preventDefault.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const mx = event.clientX - rect.left
      const my = event.clientY - rect.top
      setView((v) => {
        const k = Math.min(20, Math.max(1, v.k * Math.exp(-event.deltaY * 0.0015)))
        const ratio = k / v.k
        // Keep the cursor anchored: solve for the translation that fixes (mx,my).
        return { k, x: mx - (mx - v.x) * ratio, y: my - (my - v.y) * ratio }
      })
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
    // Re-run once points arrive: on the first mount the canvas does not exist yet
    // (the loading branch returns early), so the listener must attach after data.
  }, [points.length])

  const resetView = () => setView({ k: 1, x: 0, y: 0 })

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
      // While a search or cluster is active, dimmed points are not hoverable.
      if (focus && !focus(point)) continue
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
        style={{ width: size.width, height: size.height, cursor: dragging ? 'grabbing' : 'crosshair' }}
        onMouseDown={(event) => {
          drag.current = { x: event.clientX, y: event.clientY, moved: false }
          setDragging(true)
        }}
        onMouseMove={(event) => {
          if (drag.current) {
            const dx = event.clientX - drag.current.x
            const dy = event.clientY - drag.current.y
            if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true
            drag.current.x = event.clientX
            drag.current.y = event.clientY
            setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }))
            setHovered(null)
          } else {
            setHovered(nearest(event.clientX, event.clientY))
          }
        }}
        onMouseUp={(event) => {
          const wasDrag = drag.current?.moved
          drag.current = null
          setDragging(false)
          if (wasDrag) return // a pan should not also select a point
          const hit = nearest(event.clientX, event.clientY)
          if (hit) onSelect(hit.point.id)
        }}
        onMouseLeave={() => {
          drag.current = null
          setDragging(false)
          setHovered(null)
        }}
        onDoubleClick={resetView}
      />

      {hovered && (
        <div
          className="map__tooltip"
          style={{
            // TIP_W matches the tooltip's CSS width; TIP_H is a conservative
            // estimate of its (variable) height so it never spills off an edge.
            left: Math.min(hovered.x + 14, size.width - TIP_W - 8),
            top: Math.min(hovered.y + 14, Math.max(8, size.height - TIP_H)),
          }}
        >
          <img
            src={`/media/thumbs/${hovered.point.id}.jpg`}
            alt={hovered.point.id}
            onError={(e) => {
              e.currentTarget.style.visibility = 'hidden'
            }}
          />
          <span className="map__tooltip-label">
            <span
              className="map__swatch"
              style={{ background: colorFor(hovered.point.cluster) }}
            />
            {labelFor(hovered.point.cluster)}
          </span>
          <span className="map__tooltip-id">{hovered.point.id}</span>
        </div>
      )}

      <div className="map__panel">
        <strong>Map of the dataset in CLIP's eyes</strong>
        <p className="map__help">
          Each dot is one image, placed by UMAP so that images CLIP finds visually
          similar sit close together. What matters is <em>proximity</em>, not the
          axes: near points look alike, the axis values themselves mean nothing.
          Colours are the {clusters.length} clusters below.
        </p>
        <p className="map__help">
          Scroll to zoom, drag to pan, double-click to reset. Hover a dot for a
          preview, click to open it. Hover a theme below to isolate it.
        </p>
        {searching && (
          <p className="map__help map__help--accent">
            {highlightIds.size} search hits highlighted, everything else dimmed.
          </p>
        )}
        {view.k > 1.05 && (
          <button type="button" className="map__reset" onClick={resetView}>
            Reset view ({view.k.toFixed(1)}×)
          </button>
        )}
        <ul
          className="map__legend"
          onMouseLeave={() => setActiveCluster(null)}
          aria-label="Clusters, hover to isolate"
        >
          {clusters.map((cluster) => (
            <li
              key={cluster.id}
              className={
                'map__legend-row' +
                (activeCluster === cluster.id ? ' map__legend-row--active' : '')
              }
              onMouseEnter={() => setActiveCluster(cluster.id)}
            >
              <span className="map__swatch" style={{ background: colorFor(cluster.id) }} />
              <span className="map__legend-name">{cluster.label}</span>
              <span className="map__legend-count">{cluster.size.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
