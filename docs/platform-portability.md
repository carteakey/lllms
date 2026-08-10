# Platform portability decisions

The canonical repository remains `carteakey/l3ms`; renaming the remote is an
external GitHub administration action and is tracked in CAR-146. Local code
and documentation should continue using the existing remote until that action
is intentionally completed.

The Rust application is a native terminal binary. Its process supervision,
`pgrep`/`ps` detection, and signal-based shutdown are intentionally guarded by
platform-specific implementations. A WebAssembly/browser port remains
speculative: it would need a different process, filesystem, and llama-swap
transport boundary, so it is not part of the current release path.

When portability work is chosen, preserve the Rust/Python compatibility boundary
and add a separate target-specific issue rather than duplicating the backlog in
this document.
