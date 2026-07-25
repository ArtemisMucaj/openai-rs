# openai-rs — Agent & Contributor Guide

Canonical reference for anyone (human or AI agent) working on this crate.

---

## Project Overview

A client for OpenAI-compatible servers. It speaks two wire protocols — the
Responses API and Chat Completions — and arbitrates between them, plus model
discovery and embeddings.

Two constraints shape everything here:

1. **The Responses API is primary.** `/chat/completions` is the fallback, not
   the default. The discovered choice is cached per client.
2. **No configuration resolution.** This crate reads no environment variables
   and touches no files. Callers resolve credentials themselves and pass an
   `Endpoint`. Anything that looks like config loading belongs in the host.

---

## Architecture

Domain-Driven Design with strict Ports & Adapters layering. Dependencies always
point inward.

```
┌──────────────────────────────────────────────┐
│  Connector  (src/connector)                  │
│  • Transport (reqwest)                       │
│  • Protocol modules: responses, chat         │
│  • Adapters implementing the ports           │
└───────────────────────┬──────────────────────┘
                        │ implements
┌───────────────────────▼──────────────────────┐
│  Application  (src/application)              │
│  • Port traits only — no implementations     │
└───────────────────────┬──────────────────────┘
                        │ depends on
┌───────────────────────▼──────────────────────┐
│  Domain  (src/domain)                        │
│  • Value types, error enum                   │
│  • No I/O, no async, no reqwest              │
└──────────────────────────────────────────────┘
```

| Layer | Path | Responsibility |
|---|---|---|
| Domain | `src/domain/` | Pure value types (`Endpoint`, `Message`, `ChatRequest`, `Model`, `EmbeddingOptions`, `OpenAiError`). Depends on nothing beyond `serde` and `thiserror`. |
| Application | `src/application/` | Port traits (`ChatClient`, `ModelCatalog`, `EmbeddingClient`). Depends only on Domain. |
| Connector | `src/connector/` | HTTP adapters, `Transport`, wire types, SSE framing. Depends on Application + Domain. |

### Key files

| Concern | File |
|---|---|
| Protocol arbitration + flavor caching | `src/connector/adapter/openai_chat_client.rs` |
| Responses API wire types and calls | `src/connector/adapter/responses.rs` |
| Chat Completions wire types and calls | `src/connector/adapter/chat_completions.rs` |
| Signals shared by both protocols | `src/connector/adapter/protocol.rs` |
| SSE line framing | `src/connector/adapter/sse.rs` |
| HTTP carrier, auth headers, secret masking | `src/connector/adapter/transport.rs` |

### The protocol boundary

Each protocol module exposes one `execute(transport, model, request, sink)`
covering both streaming and buffered calls — the bodies differ only by a
`stream` flag. It reports failures it cannot resolve locally as a
`ProtocolError`:

| Variant | Meaning | Who handles it |
|---|---|---|
| `WrongApi(error)` | This model belongs to the other API, or the endpoint does not exist here | `OpenAiChatClient` retries on the other protocol |
| `SchemaUnsupported` | The server rejected the structured-output constraint | `OpenAiChatClient` retries unconstrained |
| `Fatal(error)` | Anything else | Propagated |

**`WrongApi` must only ever be returned before a single token is emitted.** That
invariant is what makes retrying a *stream* on the other protocol safe, and it
holds because the signal is read from the response status before the body is
consumed. Do not move that check.

---

## Build, Run & Test

```bash
cargo build
cargo test
cargo fmt && cargo clippy --all-targets
```

### Test layout

| Suite | What it covers |
|---|---|
| `#[cfg(test)]` modules | Wire-format parsing, serialization shape, policy decisions, pure helpers |
| `tests/chat_tests.rs` | Which endpoint is hit, how often, and in what order |
| `tests/streaming_tests.rs` | Token delivery, fallback without replay, `[DONE]` handling |
| `tests/catalog_and_embedding_tests.rs` | Model listing, batching, ordering, normalisation, credentials |

**No network in tests.** Integration tests run against a `wiremock` server bound
to localhost. Nothing may call a real provider.

Assertions on *request counts* are load-bearing — they are how flavor caching
and the batch splitter are verified. Do not relax them into "at least one".

---

## Code Style

- Follow `cargo fmt` (rustfmt defaults) and fix everything `cargo clippy` flags.
- Prefer `?` for propagation. No `.unwrap()` or `.expect()` in library code; in
  tests `.unwrap()` is fine when failure should panic immediately.
- Avoid `clone()` where borrowing suffices. `ChatRequest` is passed by reference
  precisely so a retry never re-allocates the prompt.
- All I/O is `async`. Port traits use `#[async_trait]`.
- `PascalCase` types, `snake_case` functions, `SCREAMING_SNAKE_CASE` constants.

### Error handling

- `OpenAiError` lives in `src/domain/error.rs` and must stay free of `reqwest`
  types — swapping transports should not ripple into callers' `match` arms.
- Do not swallow errors. If one must be dropped, log it with `tracing::warn!`
  first.
- When a response body explains a failure, include it in the error. A bare
  status code is not diagnosable.

### Logging

- Use `tracing` macros. Never `println!`/`eprintln!` in library code.
- **Never log a credential.** `transport::mask_secret` exists for this; the
  `Authorization` header is marked sensitive so `reqwest` will not print it
  either.

### No magic numbers

Every numeric constant gets a name and a comment explaining the choice — retry
counts, backoffs, batch sizes, timeouts.

### Module organisation

One logical concept per file; split past ~300 lines. Wire types live beside the
protocol that uses them, never in the domain.

---

## Adding to this crate

### A new capability (e.g. audio, moderation)

1. Add the value types to `src/domain/`.
2. Add a route to `ApiRoutes` — do not hardcode a path in the adapter, or
   providers with a different layout will break.
3. Define the port in `src/application/interfaces/`.
4. Implement the adapter in `src/connector/adapter/`, built on `Transport`.
5. Re-export from the layer `mod.rs` files and `lib.rs`.
6. Cover the wire format with unit tests and the HTTP behaviour with a
   `wiremock` suite.

### A new provider quirk

Prefer expressing it as **data on `Endpoint`** (a route, a header) over a
branch in the protocol modules. That is what lets a provider like Copilot reuse
this crate without a single conditional here.

---

## Commit Style

[Conventional Commits](https://www.conventionalcommits.org/):
`feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `ci`.

- Imperative mood ("add", not "added"), subject under 72 characters.
- Mark breaking changes with `!` or a `BREAKING CHANGE:` footer.

---

## No References to Private Codebases

This crate is developed alongside private repositories. **Never** include real
repository names, namespaces, class or file paths, or symbol names from them in
code, comments, tests, commit messages, or docs. Use generic placeholders
(`some-model`, `api.example.com`, `acme`).
