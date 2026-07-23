import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import type { ImageSummary, SearchMode } from './types'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

/** Run an async function on mount and whenever `deps` change, with cancellation. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: null, loading: true, error: null })

  useEffect(() => {
    let cancelled = false
    setState((previous) => ({ ...previous, loading: true, error: null }))
    fn()
      .then((data) => !cancelled && setState({ data, loading: false, error: null }))
      .catch((error: Error) => !cancelled && setState({ data: null, loading: false, error: error.message }))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

const PAGE_SIZE = 60

export interface Gallery {
  items: ImageSummary[]
  total: number
  loading: boolean
  error: string | null
  hasMore: boolean
  loadMore: () => void
  /** True when the list is a ranked result set rather than the raw dataset order. */
  ranked: boolean
}

/**
 * Drives the main image list.
 *
 * Browsing is paginated and appends on scroll; searching returns a single ranked
 * page, because past the first few dozen hits a ranked list stops being useful.
 */
export function useGallery(query: string, mode: SearchMode, split: string | undefined): Gallery {
  const [items, setItems] = useState<ImageSummary[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const requestId = useRef(0)

  const trimmed = query.trim()
  const key = `${trimmed}|${mode}|${split}`
  const prevKey = useRef(key)

  useEffect(() => {
    // When the query, mode or split changes, start fresh from offset 0 instead of
    // fetching with the stale offset. Resetting offset re-runs this effect, and the
    // offset-0 run does the actual fetch, so exactly one request fires per change.
    if (prevKey.current !== key) {
      prevKey.current = key
      setItems([])
      setTotal(0)
      if (offset !== 0) {
        // Enter the loading state now: resetting the offset re-runs this effect,
        // but until it does the render has empty items and stale loading:false,
        // which would briefly flash the "no results" empty state.
        setLoading(true)
        setError(null)
        setOffset(0)
        return
      }
    }

    const id = ++requestId.current
    let cancelled = false
    setLoading(true)
    setError(null)

    const request = trimmed
      ? api.search({ q: trimmed, mode, split }).then((r) => ({ items: r.items, total: r.total }))
      : api
          .images({ split, offset, limit: PAGE_SIZE })
          .then((page) => ({ items: page.items, total: page.total }))

    request
      .then((result) => {
        if (cancelled || id !== requestId.current) return
        setItems((previous) => (offset === 0 || trimmed ? result.items : [...previous, ...result.items]))
        setTotal(result.total)
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trimmed, mode, split, offset])

  const loadMore = useCallback(() => {
    setOffset((previous) => previous + PAGE_SIZE)
  }, [])

  return {
    items,
    total,
    loading,
    error,
    hasMore: !trimmed && items.length < total,
    loadMore,
    ranked: Boolean(trimmed),
  }
}

/** Calls `onVisible` when the returned ref enters the viewport. */
export function useOnVisible(onVisible: () => void, enabled: boolean) {
  const ref = useRef<HTMLDivElement | null>(null)
  const callback = useRef(onVisible)
  callback.current = onVisible

  useEffect(() => {
    const node = ref.current
    if (!node || !enabled) return
    const observer = new IntersectionObserver(
      (entries) => entries[0].isIntersecting && callback.current(),
      { rootMargin: '600px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [enabled])

  return ref
}

/** Debounce a fast-changing value (used to avoid a request per keystroke). */
export function useDebounced<T>(value: T, delay = 350): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}
