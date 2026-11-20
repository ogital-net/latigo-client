# latigo-client

An opinionated `jasonrpc::client::Transport` for Latigo-owned services connecting
to the platform's custom-port mTLS listener through HAProxy. TLS uses
`tokio-aws-lc`; netstring framing, multiplexing, and response correlation use
`jasonrpc` directly over each authenticated connection.
Plain TCP is available as an explicit opt-in for tests without PKI.

Each service owns its RPC method names, payload schemas, domain errors, deadlines,
and application retry decisions. This crate supplies connection mechanics, not
business logic, authorization, subscriptions, or server-initiated RPC. HAProxy's
PROXY v2 header belongs on its backend connection, not on this client connection.

## Transport policy

- Verify the server certificate and hostname, and present a client certificate
  by default. Plaintext requires an explicit `Config::plaintext` configuration;
  TLS failures never trigger a downgrade to plaintext.
- Always enable TCP_NODELAY and TCP keepalive: 30 seconds idle, 10 seconds between
  probes, and three probes. Actual failure detection depends on the operating
  system and network. Fail setup if required socket options cannot be applied.
- Resolve A and AAAA records through the system resolver on every connection
  round and try the addresses in a fresh random order. TLS verifies the configured
  hostname, not the selected IP address.
- Limit DNS resolution, TCP connection, and TLS negotiation to ten seconds total.
  Each address gets a share of the remaining time, so failed or stalled TCP or
  TLS setup can fall back to another address before entering backoff.
- Reconnect after setup errors, EOF, or stream I/O errors. Retry indefinitely with
  exponential backoff starting at 250 ms and capped at 30 seconds, jittered by up
  to 50 percent without exceeding the cap. Reset backoff after 30 seconds connected.
- Use one shared multiplexed connection, with a background framed reader and a
  writer that serializes complete netstring frames without interleaving bytes.
- Default the incoming response frame limit to 16 MiB, including netstring
  framing; configure it with `Config::with_max_response_frame_size`. Outgoing
  requests and notifications have no client-side size cap; the server owns
  request-size policy.
- Allow at most 128 in-flight operations across all transport clones. When full, wait
  up to one second for capacity before returning `TransportError::Busy`;
  configure this with `Config::with_capacity_timeout`. Busy operations send nothing.
- Fail fast with `TransportError::Disconnected` while there is no live session.
  No requests or notifications are queued across outages.
- Reconnect proactively when a session closes or becomes invalid, even if idle.
  Pending calls fail instead of migrating to a replacement session. Requests and
  notifications are never automatically replayed.
- Keep the same outer transport and generic client usable across reconnections.
  Dropping one clone does not stop other clones. Dropping the last clone stops
  the supervisor; `shutdown()` explicitly stops all clones and pending operations.

Certificate and key loading remains with the service. `Config::new` accepts
client certificate chains as slices of DER certificate bytes and a DER private
key (PKCS#8, PKCS#1, or SEC1). Server certificates are verified using AWS-LC's
system-default CA files and directories; callers do not need to supply roots.
These are OpenSSL-style verification paths, not direct macOS Keychain access.
Ensure the deployment has a CA bundle installed; `SSL_CERT_FILE` and
`SSL_CERT_DIR` can configure its location. For a private server CA or isolated
tests, `Config::with_root_certificates` uses only the supplied DER server roots.

Server trust roots and the CA that issues client identities are separate concerns.
Credentials are fixed for a transport's lifetime; restart it with a new
configuration to rotate them. TLS is provided by `tokio-aws-lc` using AWS-LC's
native TLS engine; no insecure verification option is exposed. Linux kTLS offload
is disabled because this release does not support post-attach TLS 1.3 session
tickets or key updates; userspace TLS retains compatibility with HAProxy's
post-handshake messages.

## Connection logging

The transport emits events through the `log` crate under the `latigo_client`
target. The application installs and configures its own logger. `info` reports
established connections and their selected peer addresses. `warn` reports
lost connections and the first failure in a retry sequence, including the retry
delay. Enable `debug` for individual address attempts, TCP and TLS setup failures,
timeouts, subsequent failed rounds, and shutdown. Connection events do not
include credentials or RPC payloads.

## Plaintext testing

Use `Config::plaintext(host, port)` with a test server that speaks JSON-RPC
netstrings directly over TCP. No certificates, private keys, or CA roots are
required. This disables TLS entirely, not just certificate verification, so a
TLS-only listener cannot be used in this mode.

There is no encryption or peer authentication. Use loopback or an isolated,
trusted test network. `Config::new` and `Config::with_root_certificates` still
require client credentials and always use verified mTLS.

```rust,no_run
use latigo_client::{Client, Config, Transport};
use std::time::Duration;

async fn test_client(port: u16) -> Result<Client, Box<dyn std::error::Error + Send + Sync>> {
  let config = Config::plaintext("127.0.0.1", port)?;
  let transport = Transport::start(config);
  tokio::time::timeout(Duration::from_secs(5), transport.wait_connected()).await??;
  Ok(Client::new(transport).with_request_timeout(Duration::from_secs(2)))
}
```

Plaintext uses the same response frame limit, in-flight operation limit,
connection deadline, reconnection policy, and shutdown behavior as TLS. The
response frame limit can be set with `with_max_response_frame_size` in either mode.

## Capacity waiting

Calls and notifications wait for an available in-flight slot for up to one second
by default. Set the wait on `Config` before starting the transport; zero restores
immediate `TransportError::Busy` errors when full.

```rust,no_run
use latigo_client::Config;
use std::time::Duration;

fn allow_bursts(config: Config) -> Config {
  config.with_capacity_timeout(Duration::from_millis(500))
}
```

Nothing is sent while waiting. Shutdown returns `TransportError::Stopped`, and
loss of the current session returns `TransportError::Disconnected`; waiting
operations do not move to a replacement session. Cancelling a wait does not
consume a slot. Waiting calls retain their unsent payloads, so callers should
still bound concurrency to control memory use.

## Response frame limit

Set `with_max_response_frame_size` on `Config` before starting the transport to
configure the multiplexed response reader. The setting persists across
reconnections and defaults to 16 MiB. The limit counts encoded JSON plus netstring
framing, not the size of an unencoded image. Oversized responses invalidate the
session, fail pending calls, and trigger reconnection.

```rust,no_run
use latigo_client::Config;

fn configure_large_responses(config: Config) -> Config {
  config.with_max_response_frame_size(80 * 1024 * 1024)
}
```

Zero disables the incoming cap. Prefer a finite limit to bound client-side
buffering, even on authenticated connections.

Requests and notifications are not size-limited by this client. The server must
permit the intended encoded request size; the response limit does not negotiate
server limits. Large images encoded as base64 grow by roughly one third before
the JSON envelope is added.

## Service integration

```rust,no_run
use latigo_client::{Client, Config, Transport, jasonrpc::error::ClientError};
use std::{sync::Arc, time::Duration};

struct ServiceClient {
    rpc: Arc<Client>,
}

impl ServiceClient {
    async fn lookup(&self, key: &str) -> Result<String, ClientError> {
        self.rpc.call("service.lookup", (key,)).await
    }
}

async fn run(config: Config) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let transport = Transport::start(config);
    tokio::time::timeout(Duration::from_secs(15), transport.wait_connected()).await??;

    let rpc = Arc::new(
        Client::new(transport.clone()).with_request_timeout(Duration::from_secs(2)),
    );
    let service = ServiceClient { rpc };
    let _result = service.lookup("example").await?;

    transport.shutdown();
    Ok(())
}
```

`Client` is an alias for `jasonrpc::client::Client<Transport>`. Share one generic
client through `Arc` across service wrappers and concurrent callers so they use
one request-ID allocator. Independent generic clients over transport clones
have independent allocators and can collide; `jasonrpc` rejects duplicate
in-flight IDs. The example method and result type belong to the service, not
this library.

Notifications use the send-only transport method and do not wait for a response.
Successful notification submission does not acknowledge application execution.
Connection loss, cancellation, or shutdown during an operation can leave its
outcome unknown. Services must decide whether an operation can be retried and
provide any required idempotency or deduplication. A response-wait cancellation
leaves the session usable; cancellation during a partial frame write invalidates
the session and triggers reconnection.

`wait_connected()` and `ConnectionState::Connected` report transport connectivity,
not application readiness. Status subscriptions contain the latest state and
may skip intermediate transitions. Dial failures are reported by
`ConnectionState::BackingOff`; established-session closure is reported generically
because `jasonrpc` supplies a closure signal rather than its reader's error.
Use a Tokio runtime with I/O and time enabled.

## Upstream dependency

This checkout temporarily uses a `[patch.crates-io]` override pointing to the
upstream `jasonrpc` repository's `main` branch. The required APIs were merged in
[commit f42c8d6](https://github.com/ogital-net/jasonrpc/commit/f42c8d6002fa39d46c5b65314e165e83d62ad90b)
and will be included in the next release.

`MultiplexTransport::closed()` provides an awaitable notification for EOF,
framing/I/O failure, and cancelled partial writes. It supports concurrent waiters
and returns immediately for an already-invalid session.
`MultiplexTransport::new_with_max_frame_size()` configures the incoming limit
before its reader task starts, preserving `new()`'s existing 16 MiB default.

The override is required until those APIs are published in a compatible `jasonrpc`
release. When consuming this checkout as a path dependency, put the same override
in the consuming application's root manifest: dependency-local patches are not
inherited. After the upstream release is available, remove the Git override
and require the release containing the new APIs.

## Checks

```sh
cargo test
cargo check --lib
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```