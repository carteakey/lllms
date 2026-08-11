# Platform portability decisions

The canonical GitHub remote is already `carteakey/l3ms`; `origin` points to
`https://github.com/carteakey/l3ms.git`. No repository rename is currently
required. Local code and documentation should continue using this remote until
a future issue names a specific replacement and the GitHub administration
change is intentionally approved.

The Rust application is a native terminal binary. Its process supervision,
`pgrep`/`ps` detection, and signal-based shutdown are intentionally guarded by
platform-specific implementations. A WebAssembly/browser port remains
speculative: it would need a different process, filesystem, and llama-swap
transport boundary, so it is not part of the current release path.

When portability work is chosen, preserve the Rust/Python compatibility boundary
and add a separate target-specific issue rather than duplicating the backlog in
this document.
