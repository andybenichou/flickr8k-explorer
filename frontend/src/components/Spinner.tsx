interface Props {
  /** Text shown next to the spinner. */
  label?: string
  /** Adds vertical padding so a standalone loader is not cramped against content. */
  block?: boolean
}

/** The single loading indicator used everywhere, so every wait looks the same. */
export function Spinner({ label = 'Loading…', block = false }: Props) {
  return (
    <p className={'notice notice--loading' + (block ? ' notice--block' : '')} role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label}
    </p>
  )
}
