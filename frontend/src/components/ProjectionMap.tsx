import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { Projection, ProjectionPoint } from '../types'
import { Spinner } from './Spinner'

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
  // The canvas and its container are held as state, not refs, because the first
  // load returns from the loading branch below before either node exists. State
  // re-runs the effects that draw and that attach the observer/listeners once the
  // nodes are mounted, where a ref mutation would leave them attached to nothing.
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)
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
  // Keyboard "cursor": the point the arrow keys are currently on. Pointer users
  // hover; this is the keyboard-and-screen-reader equivalent, drawn as a dashed
  // ring and announced via the live region below.
  const [kbFocusId, setKbFocusId] = useState<string | null>(null)
  // Collapsible overlay panel. Default open (the desktop layout has room for it),
  // then collapse on a phone, where it otherwise blankets most of the map. The
  // decision is made in an effect below so it reads the settled viewport rather
  // than a mount-time width that can be stale under this lazy-mounted view.
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

  // Collapse the panel by default on a phone, where it otherwise blankets the
  // map. Only collapse for a definite narrow width: an unsettled 0 (this view
  // mounts lazily, before layout resolves) falls through to the open default,
  // which is the right choice for the desktop layout.
  useEffect(() => {
    const w = window.innerWidth
    if (w > 0 && w <= 640) setPanelOpen(false)
  }, [])

  // Track the container size so the map fills the available space. Keyed on the
  // container node itself, so it observes the node as soon as it mounts. Zero
  // measurements are ignored: a detached node and a display:none one (this view
  // while another tab is showing) both report 0x0, and adopting that would size the
  // canvas to nothing and blank the map with no later resize to recover from.
  useEffect(() => {
    if (!container) return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) setSize({ width, height })
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [container])

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

    // Dashed accent ring for the keyboard cursor, under the solid selection ring.
    const kbPoint = kbFocusId ? points.find((p) => p.id === kbFocusId) : null
    if (kbPoint) {
      const { x, y } = toPixel(kbPoint)
      ctx.globalAlpha = 1
      ctx.strokeStyle = '#7dd3fc'
      ctx.lineWidth = 2
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.stroke()
      ctx.setLineDash([])
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
    // `canvas` is a dependency so the freshly mounted, still-blank canvas is drawn
    // on the render where it appears, without waiting for a data or viewport change.
  }, [canvas, points, size, toPixel, focus, selectedId, view, kbFocusId])

  // Wheel zoom, centred on the cursor so the point under it stays put. Attached
  // manually because React's onWheel is passive and cannot preventDefault.
  useEffect(() => {
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
    // Keyed on the node: the first load returns from the loading branch before the
    // canvas exists, so the listener has to attach once it is actually mounted.
  }, [canvas])

  const resetView = () => setView({ k: 1, x: 0, y: 0 })

  /** Zoom by a factor around the canvas centre (keyboard +/- zoom). */
  const zoomBy = (factor: number) =>
    setView((v) => {
      const k = Math.min(20, Math.max(1, v.k * factor))
      const ratio = k / v.k
      const cx = size.width / 2
      const cy = size.height / 2
      return { k, x: cx - (cx - v.x) * ratio, y: cy - (cy - v.y) * ratio }
    })

  /**
   * Move the keyboard cursor to the nearest point in a direction. Works in data
   * space so it is independent of the current pan/zoom: `along` is the distance
   * in the pressed direction (must be positive), `off` the perpendicular drift,
   * penalised so the pick stays roughly on-axis. Dimmed points are skipped, matching
   * the pointer's nearest().
   */
  const stepFocus = (dir: 'left' | 'right' | 'up' | 'down') => {
    if (points.length === 0) return
    const from = kbFocusId ? points.find((p) => p.id === kbFocusId) : null
    if (!from) {
      // First key press: start on the selected point, else the first one.
      const start = points.find((p) => p.id === selectedId) ?? points[0]
      setKbFocusId(start.id)
      return
    }
    let best: ProjectionPoint | null = null
    let bestCost = Infinity
    for (const point of points) {
      if (point.id === from.id) continue
      if (focus && !focus(point)) continue
      const dx = point.x - from.x
      const dy = point.y - from.y
      let along: number
      let off: number
      if (dir === 'right') { along = dx; off = Math.abs(dy) }
      else if (dir === 'left') { along = -dx; off = Math.abs(dy) }
      else if (dir === 'up') { along = dy; off = Math.abs(dx) }
      else { along = -dy; off = Math.abs(dx) }
      if (along <= 1e-6) continue
      const cost = along + off * 2.5
      if (cost < bestCost) {
        bestCost = cost
        best = point
      }
    }
    if (best) setKbFocusId(best.id)
  }

  // Keep the keyboard cursor on screen: if the focused point sits outside the
  // visible canvas (e.g. after stepping while zoomed in), pan to centre it.
  useEffect(() => {
    if (!kbFocusId) return
    const point = points.find((p) => p.id === kbFocusId)
    if (!point) return
    const { x, y } = toPixel(point)
    const margin = 44
    if (x < margin || y < margin || x > size.width - margin || y > size.height - margin) {
      const b = toBase(point)
      setView((v) => ({ ...v, x: size.width / 2 - b.x * v.k, y: size.height / 2 - b.y * v.k }))
    }
    // Only react to the cursor moving; view/size are read as current values.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbFocusId])

  /** Nearest point to the cursor, within a small radius. 8k points is a cheap scan. */
  function nearest(clientX: number, clientY: number) {
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

  // Only the very first load blanks the view: a later split change keeps the
  // current map in place under a scrim (see `.map__loading` below), so the pane
  // never collapses to a centred spinner and back.
  if (projection.loading && !projection.data) return <Spinner block label="Loading projection…" />
  if (projection.error) return <p className="notice notice--error" role="alert">{projection.error}</p>
  if (points.length === 0)
    return (
      <p className="notice">
        No 2D projection stored. Re-run the ingestion without <code>--skip-projection</code>.
      </p>
    )

  return (
    <div className="map" ref={setContainer}>
      <canvas
        ref={setCanvas}
        // Focusable interactive widget: role="application" so a screen reader
        // passes the arrow/Enter keys through to our handler instead of eating
        // them for its own navigation. Live updates go to the region below.
        role="application"
        tabIndex={0}
        aria-label={`Scatter plot of ${points.length.toLocaleString()} images placed by visual similarity in ${clusters.length} colour-coded clusters. Arrow keys move between points, Enter opens the focused image, plus and minus zoom, Home resets the view.`}
        onFocus={() => {
          if (!kbFocusId) {
            const start = points.find((p) => p.id === selectedId) ?? points[0]
            if (start) setKbFocusId(start.id)
          }
        }}
        aria-busy={projection.loading || undefined}
        onKeyDown={(event) => {
          // The scrim blocks the pointer; the keyboard needs the same guard so
          // Enter cannot open a point from the split being replaced.
          if (projection.loading) return
          switch (event.key) {
            case 'ArrowRight': event.preventDefault(); stepFocus('right'); break
            case 'ArrowLeft': event.preventDefault(); stepFocus('left'); break
            case 'ArrowUp': event.preventDefault(); stepFocus('up'); break
            case 'ArrowDown': event.preventDefault(); stepFocus('down'); break
            case 'Enter':
            case ' ':
              if (kbFocusId) { event.preventDefault(); onSelect(kbFocusId) }
              break
            case 'Home':
            case '0': event.preventDefault(); resetView(); break
            case '+':
            case '=': event.preventDefault(); zoomBy(1.3); break
            case '-':
            case '_': event.preventDefault(); zoomBy(1 / 1.3); break
            case 'Escape': setKbFocusId(null); break
          }
        }}
        // touchAction:'none' stops the browser hijacking the gesture for page
        // scroll/zoom, so our own touch handlers own every finger on the canvas.
        style={{ width: size.width, height: size.height, cursor: dragging ? 'grabbing' : 'crosshair', touchAction: 'none' }}
        onTouchStart={(event) => {
          // Touch has no hover, so never leave a stale tooltip on screen.
          setHovered(null)
          if (event.touches.length === 1) {
            const t = event.touches[0]
            touch.current = { mode: 'pan', x: t.clientX, y: t.clientY, moved: false, distance: 0 }
          } else if (event.touches.length >= 2) {
            const [a, b] = [event.touches[0], event.touches[1]]
            const distance = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)
            // moved:true so lifting out of a pinch never counts as a tap-select.
            touch.current = { mode: 'pinch', x: 0, y: 0, moved: true, distance }
          }
        }}
        onTouchMove={(event) => {
          const state = touch.current
          if (!state) return
          if (state.mode === 'pan' && event.touches.length === 1) {
            // Same delta-translate math as the mouse drag.
            const t = event.touches[0]
            const dx = t.clientX - state.x
            const dy = t.clientY - state.y
            if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true
            state.x = t.clientX
            state.y = t.clientY
            setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }))
          } else if (state.mode === 'pinch' && event.touches.length >= 2) {
            const [a, b] = [event.touches[0], event.touches[1]]
            const distance = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)
            const prev = state.distance || distance
            state.distance = distance
            if (!canvas) return
            const rect = canvas.getBoundingClientRect()
            // Zoom centred on the midpoint between the two fingers.
            const mx = (a.clientX + b.clientX) / 2 - rect.left
            const my = (a.clientY + b.clientY) / 2 - rect.top
            setView((v) => {
              const k = Math.min(20, Math.max(1, v.k * (distance / prev)))
              const ratio = k / v.k
              // Keep the focal point anchored, exactly like the wheel handler.
              return { k, x: mx - (mx - v.x) * ratio, y: my - (my - v.y) * ratio }
            })
          }
        }}
        onTouchEnd={(event) => {
          const state = touch.current
          const remaining = event.touches.length
          // A one-finger tap that did not move selects the nearest point, and a
          // quick second tap resets the view (double-tap).
          if (state && state.mode === 'pan' && remaining === 0 && !state.moved) {
            const t = event.changedTouches[0]
            const now = Date.now()
            if (now - lastTap.current < 300) {
              resetView()
              lastTap.current = 0
            } else {
              lastTap.current = now
              const hit = nearest(t.clientX, t.clientY)
              if (hit) onSelect(hit.point.id)
            }
          }
          if (remaining === 0) {
            touch.current = null
          } else if (remaining === 1) {
            // Lifting one finger out of a pinch: continue as a pan, but with
            // moved:true so releasing the last finger does not select a point.
            const t = event.touches[0]
            touch.current = { mode: 'pan', x: t.clientX, y: t.clientY, moved: true, distance: 0 }
          }
        }}
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

      {/* Announces the keyboard cursor to assistive tech (the canvas pixels can't). */}
      <div className="sr-only" role="status" aria-live="polite">
        {kbFocusId
          ? `Focused ${kbFocusId}, ${labelFor(points.find((p) => p.id === kbFocusId)?.cluster ?? null)}. Press Enter to open.`
          : ''}
      </div>

      {hovered && !projection.loading && (
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
              // Collapse (not just hide) so the tooltip shrinks cleanly instead
              // of leaving a blank gap above the label.
              e.currentTarget.style.display = 'none'
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

      <div className={'map__panel' + (panelOpen ? '' : ' map__panel--collapsed')}>
        <strong>Map of the dataset in CLIP's eyes</strong>
        <button
          type="button"
          className="map__panel-toggle"
          onClick={() => setPanelOpen((open) => !open)}
          aria-expanded={panelOpen}
        >
          {panelOpen ? 'Hide' : 'Show'}
        </button>
        <p className="map__help">
          Each dot is one image, placed by UMAP so that images CLIP finds visually
          similar sit close together. What matters is <em>proximity</em>, not the
          axes: near points look alike, the axis values themselves mean nothing.
          Colours are the {clusters.length} clusters below.
        </p>
        <p className="map__help">
          Scroll to zoom, drag to pan, double-click to reset. Hover a dot for a
          preview, click to open it. Hover a theme below to isolate it. Or focus
          the map and use the arrow keys to walk between points, Enter to open.
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
              // Not a native button (it lives in a <ul>), so wire up the button
              // role, focusability and Enter/Space so it is keyboard-operable.
              role="button"
              tabIndex={0}
              aria-pressed={activeCluster === cluster.id}
              onMouseEnter={() => setActiveCluster(cluster.id)}
              // Tap to isolate on touch (no hover); tap again to clear.
              onClick={() => setActiveCluster((c) => (c === cluster.id ? null : cluster.id))}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  setActiveCluster((c) => (c === cluster.id ? null : cluster.id))
                }
              }}
            >
              <span className="map__swatch" style={{ background: colorFor(cluster.id) }} />
              <span className="map__legend-name">{cluster.label}</span>
              <span className="map__legend-count">{cluster.size.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Covers the whole map while the next split loads: the points underneath
          belong to the previous split, so they must not be clickable. */}
      {projection.loading && (
        <div className="map__loading">
          <Spinner label="Loading projection…" />
        </div>
      )}
    </div>
  )
}
