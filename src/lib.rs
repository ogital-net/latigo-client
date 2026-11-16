#![doc = include_str!("../README.md")]

use std::{
    future::Future,
    io,
    sync::{Arc, Mutex},
    time::Duration,
};

use jasonrpc::transport::{Netstring, io::DEFAULT_MAX_FRAME_SIZE};
use jasonrpc::{
    Id,
    client::{
        MultiplexError, MultiplexOver, MultiplexTransport, ReceivedResponse,
        Transport as RpcTransport,
    },
};
use socket2::{SockRef, TcpKeepalive};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
    sync::{Semaphore, SemaphorePermit, watch},
    task::JoinHandle,
    time::{Instant, sleep, timeout},
};
use tokio_aws_lc::{ClientConfig, ClientConfigBuilder, TlsConnector};

pub use jasonrpc;

/// Generic JSON-RPC client for service-owned method and payload wrappers.
pub type Client = jasonrpc::client::Client<Transport>;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const INITIAL_BACKOFF: Duration = Duration::from_millis(250);
const MAX_BACKOFF: Duration = Duration::from_secs(30);
const STABLE_CONNECTION: Duration = Duration::from_secs(30);
const MAX_IN_FLIGHT: usize = 128;
const DEFAULT_CAPACITY_TIMEOUT: Duration = Duration::from_secs(1);

trait NetworkStream: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug {}

impl<Stream: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug> NetworkStream for Stream {}

type Session = MultiplexOver<Box<dyn NetworkStream>, Netstring>;

/// Endpoint and optional verified mTLS credentials. The caller selects the port.
#[derive(Clone)]
pub struct Config {
    host: String,
    port: u16,
    tls: Option<Arc<ClientConfig>>,
    max_response_frame_size: usize,
    capacity_timeout: Duration,
}

impl Config {
    /// Use system-default server trust with the client's certificate and key.
    ///
    /// The host is also the TLS server name; DNS names and IP addresses are
    /// verified against the server certificate. Client identity certificates
    /// are separate from the roots used to trust the server.
    ///
    /// Trust uses AWS-LC/OpenSSL default CA paths, not macOS Keychain access.
    /// Client certificates are DER-encoded certificates; the client
    /// chain starts with its leaf certificate. The DER private key may use
    /// PKCS#8, PKCS#1, or SEC1 encoding. Credentials are copied during setup.
    pub fn new(
        host: impl Into<String>,
        port: u16,
        client_certificates: &[&[u8]],
        client_key: &[u8],
    ) -> io::Result<Self> {
        Self::from_tls_builder(
            host,
            port,
            ClientConfig::builder().with_system_root_certs(),
            client_certificates,
            client_key,
        )
    }

    /// Use only the supplied DER server roots instead of system-default trust.
    ///
    /// Client credentials have the same encoding and requirements as [`Self::new`].
    pub fn with_root_certificates(
        host: impl Into<String>,
        port: u16,
        server_roots: &[&[u8]],
        client_certificates: &[&[u8]],
        client_key: &[u8],
    ) -> io::Result<Self> {
        Self::from_tls_builder(
            host,
            port,
            ClientConfig::builder().with_root_certs_der_bytes(server_roots),
            client_certificates,
            client_key,
        )
    }

    /// Use unencrypted, unauthenticated TCP for tests without PKI.
    ///
    /// No certificates or keys are required. This disables TLS entirely; it
    /// does not bypass certificate verification on a TLS connection. Do not
    /// use on untrusted networks. TLS configurations never fall back to this mode.
    pub fn plaintext(host: impl Into<String>, port: u16) -> io::Result<Self> {
        let host = host.into();
        if host.is_empty() || !host.is_ascii() || host.contains('\0') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host must be a nonempty ASCII DNS name or IP address without NUL bytes",
            ));
        }
        Ok(Self {
            host,
            port,
            tls: None,
            max_response_frame_size: DEFAULT_MAX_FRAME_SIZE,
            capacity_timeout: DEFAULT_CAPACITY_TIMEOUT,
        })
    }

    /// Set how long operations wait for an in-flight slot before returning busy.
    ///
    /// Defaults to one second. Zero fails immediately when at capacity. The wait
    /// sends nothing and ends early on shutdown or loss of the current session.
    #[must_use]
    pub fn with_capacity_timeout(mut self, timeout: Duration) -> Self {
        self.capacity_timeout = timeout;
        self
    }

    /// Set the maximum incoming response frame size in bytes.
    ///
    /// Includes netstring framing and defaults to 16 MiB. Zero disables the
    /// limit. Applied to each new session, including after reconnection.
    /// Outgoing requests and notifications are not size-limited by this client.
    #[must_use]
    pub fn with_max_response_frame_size(mut self, limit: usize) -> Self {
        self.max_response_frame_size = limit;
        self
    }

    fn from_tls_builder(
        host: impl Into<String>,
        port: u16,
        builder: ClientConfigBuilder,
        client_certificates: &[&[u8]],
        client_key: &[u8],
    ) -> io::Result<Self> {
        let mut config = Self::plaintext(host, port)?;
        let tls = builder
            .with_client_auth_der_bytes(client_certificates, client_key)
            .disable_ktls()
            .build()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        config.tls = Some(Arc::new(tls));
        Ok(config)
    }
}

/// Transport status only; `Connected` does not imply application readiness.
#[derive(Clone, Debug)]
pub enum ConnectionState {
    Connecting,
    Connected,
    BackingOff {
        attempt: u32,
        delay: Duration,
        error: Arc<io::Error>,
    },
    Stopped,
}

/// A stable, cloneable JSON-RPC transport over supervised connections.
///
/// Uses verified mTLS unless explicitly configured with [`Config::plaintext`].
/// Uses netstrings and jasonrpc's multiplexing and correlation. All clones share
/// one connection and a limit of 128 in-flight operations. Calls fail fast while
/// disconnected and wait up to the configured capacity timeout when full.
/// Interrupted operations are never replayed.
#[derive(Clone)]
pub struct Transport {
    inner: Arc<Inner>,
}

#[derive(Default)]
enum SessionSlot {
    #[default]
    Vacant,
    Occupied(Session),
    Stopped,
}

struct Inner {
    session: Arc<Mutex<SessionSlot>>,
    state: watch::Sender<ConnectionState>,
    in_flight: Semaphore,
    capacity_timeout: Duration,
    task: JoinHandle<()>,
}

/// Transport failures, separate from JSON-RPC errors returned by the server.
#[derive(Debug)]
pub enum TransportError {
    Disconnected,
    Stopped,
    Busy,
    Session(MultiplexError),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => {
                formatter.write_str("transport is disconnected; nothing was sent")
            }
            Self::Stopped => formatter.write_str("transport is shut down"),
            Self::Busy => formatter.write_str("transport in-flight operation limit reached"),
            Self::Session(error) => write!(formatter, "session failed: {error}"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Session(error) => Some(error),
            _ => None,
        }
    }
}

impl Transport {
    /// Start connecting immediately, retrying failures indefinitely.
    ///
    /// DNS resolution, TCP connection, and TLS negotiation share a ten-second
    /// deadline. Backoff starts at 250 ms, doubles to 30 seconds, and is jittered
    /// by up to 50 percent without exceeding the cap. Thirty seconds connected
    /// resets backoff. Dropping the last transport clone cancels its supervisor.
    ///
    /// Must be called within a Tokio runtime with I/O and time enabled.
    pub fn start(config: Config) -> Self {
        let capacity_timeout = config.capacity_timeout;
        let session = Arc::new(Mutex::new(SessionSlot::default()));
        let (state, _) = watch::channel(ConnectionState::Connecting);
        let task = tokio::spawn(supervise(config, session.clone(), state.clone()));
        Self {
            inner: Arc::new(Inner {
                session,
                state,
                in_flight: Semaphore::new(MAX_IN_FLIGHT),
                capacity_timeout,
                task,
            }),
        }
    }

    /// Wait for transport connectivity, or return an error after shutdown.
    ///
    /// This does not establish application readiness or reserve a session for
    /// the next call. Wrap the wait in a caller-owned timeout if needed.
    pub async fn wait_connected(&self) -> Result<(), TransportError> {
        let mut state = self.subscribe();
        loop {
            {
                let slot = self.inner.session.lock().unwrap();
                match &*slot {
                    SessionSlot::Stopped => return Err(TransportError::Stopped),
                    SessionSlot::Occupied(session) if session.is_connected() => return Ok(()),
                    _ => {}
                }
            }
            state.changed().await.map_err(|_| TransportError::Stopped)?;
        }
    }

    /// Subscribe to connection transitions and the latest transport failure.
    pub fn subscribe(&self) -> watch::Receiver<ConnectionState> {
        self.inner.state.subscribe()
    }

    /// Stop all clones, cancel in-flight operations, and stop reconnecting.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    async fn session(&self) -> Result<(Session, SemaphorePermit<'_>), TransportError> {
        let mut state = self.subscribe();
        let session = {
            let slot = self.inner.session.lock().unwrap();
            match &*slot {
                SessionSlot::Stopped => return Err(TransportError::Stopped),
                SessionSlot::Occupied(session) if session.is_connected() => session.clone(),
                _ => return Err(TransportError::Disconnected),
            }
        };
        let permit = tokio::select! {
            biased;
            _ = state.wait_for(|state| matches!(state, ConnectionState::Stopped)) => {
                return Err(TransportError::Stopped);
            }
            _ = session.closed() => return Err(TransportError::Disconnected),
            result = timeout(self.inner.capacity_timeout, self.inner.in_flight.acquire()) => {
                result
                    .map_err(|_| TransportError::Busy)?
                    .map_err(|_| TransportError::Stopped)?
            }
        };
        if matches!(*self.inner.session.lock().unwrap(), SessionSlot::Stopped) {
            return Err(TransportError::Stopped);
        }
        if !session.is_connected() {
            return Err(TransportError::Disconnected);
        }
        Ok((session, permit))
    }

    async fn run<Output>(
        &self,
        session: &Session,
        operation: impl Future<Output = Result<Output, MultiplexError>>,
    ) -> Result<Output, TransportError> {
        let mut state = self.subscribe();
        tokio::select! {
            biased;
            _ = state.wait_for(|state| matches!(state, ConnectionState::Stopped)) => {
                Err(TransportError::Stopped)
            }
            result = operation => result.map_err(TransportError::Session),
            _ = session.closed() => {
                Err(TransportError::Session(MultiplexError::ConnectionClosed))
            }
        }
    }
}

impl RpcTransport for Transport {
    type Error = TransportError;

    async fn round_trip(&self, request: Vec<u8>) -> Result<Vec<u8>, Self::Error> {
        let (session, _permit) = self.session().await?;
        self.run(&session, session.round_trip(request)).await
    }

    async fn round_trip_with_id(&self, request: Vec<u8>, id: Id) -> Result<Vec<u8>, Self::Error> {
        let (session, _permit) = self.session().await?;
        self.run(&session, session.round_trip_with_id(request, id))
            .await
    }

    async fn round_trip_response(
        &self,
        request: Vec<u8>,
        id: Id,
    ) -> Result<ReceivedResponse, Self::Error> {
        let (session, _permit) = self.session().await?;
        self.run(&session, session.round_trip_response(request, id))
            .await
    }

    async fn send_notification(&self, notification: Vec<u8>) -> Result<(), Self::Error> {
        let (session, _permit) = self.session().await?;
        self.run(&session, session.send_notification(notification))
            .await
    }
}

impl Inner {
    fn shutdown(&self) {
        let mut slot = self.session.lock().unwrap();
        *slot = SessionSlot::Stopped;
        self.state.send_replace(ConnectionState::Stopped);
        self.task.abort();
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn connect_tcp(host: &str, port: u16) -> io::Result<TcpStream> {
    let stream = TcpStream::connect((host, port)).await?;
    stream.set_nodelay(true)?;
    SockRef::from(&stream).set_tcp_keepalive(
        &TcpKeepalive::new()
            .with_time(Duration::from_secs(30))
            .with_interval(Duration::from_secs(10))
            .with_retries(3),
    )?;
    Ok(stream)
}

async fn connect(config: &Config) -> io::Result<Box<dyn NetworkStream>> {
    timeout(CONNECT_TIMEOUT, async {
        let socket = connect_tcp(&config.host, config.port).await?;
        let network: Box<dyn NetworkStream> = match &config.tls {
            Some(tls) => Box::new(
                TlsConnector::new(tls.clone())
                    .connect(&config.host, socket)
                    .await
                    .map_err(|error| match error {
                        tokio_aws_lc::Error::Io(error) => error,
                        error => io::Error::other(error),
                    })?,
            ),
            None => Box::new(socket),
        };
        Ok(network)
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "connection setup timed out"))?
}

#[derive(Default)]
struct Backoff {
    attempt: u32,
}

impl Backoff {
    fn next(&mut self) -> Duration {
        let base = INITIAL_BACKOFF
            .saturating_mul(1_u32 << self.attempt.min(7))
            .min(MAX_BACKOFF);
        self.attempt = self.attempt.saturating_add(1);
        let lower = (base.as_millis() / 2) as u64;
        let upper = (base.as_millis() * 3 / 2).min(MAX_BACKOFF.as_millis()) as u64;
        Duration::from_millis(fastrand::u64(lower..=upper))
    }
}

async fn supervise(
    config: Config,
    sessions: Arc<Mutex<SessionSlot>>,
    state: watch::Sender<ConnectionState>,
) {
    let mut backoff = Backoff::default();
    loop {
        {
            let slot = sessions.lock().unwrap();
            if matches!(*slot, SessionSlot::Stopped) {
                return;
            }
            state.send_replace(ConnectionState::Connecting);
        }
        let error = match connect(&config).await {
            Ok(network) => {
                let session = MultiplexTransport::new_with_max_frame_size(
                    network,
                    Netstring,
                    config.max_response_frame_size,
                );
                let connected_at = Instant::now();
                {
                    let mut slot = sessions.lock().unwrap();
                    if matches!(*slot, SessionSlot::Stopped) {
                        return;
                    }
                    *slot = SessionSlot::Occupied(session.clone());
                    state.send_replace(ConnectionState::Connected);
                }
                session.closed().await;
                {
                    let mut slot = sessions.lock().unwrap();
                    if matches!(*slot, SessionSlot::Stopped) {
                        return;
                    }
                    *slot = SessionSlot::Vacant;
                }
                if connected_at.elapsed() >= STABLE_CONNECTION {
                    backoff = Backoff::default();
                }
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "multiplexed session closed or invalidated",
                )
            }
            Err(error) => error,
        };
        let delay = backoff.next();
        {
            let slot = sessions.lock().unwrap();
            if matches!(*slot, SessionSlot::Stopped) {
                return;
            }
            state.send_replace(ConnectionState::BackingOff {
                attempt: backoff.attempt,
                delay,
                error: Arc::new(error),
            });
        }
        sleep(delay).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jasonrpc::{Request, Response, error::ClientError, json, transport::io::FramedConn};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_aws_lc::{ClientAuthMode, ServerConfig, TlsAcceptor};

    fn tls_config(port: u16) -> (Config, TlsAcceptor) {
        tls_config_for_host(port, "localhost", "localhost")
    }

    fn tls_config_for_host(port: u16, host: &str, certificate_name: &str) -> (Config, TlsAcceptor) {
        let server = rcgen::generate_simple_self_signed(vec![certificate_name.into()]).unwrap();
        let client = rcgen::generate_simple_self_signed(vec!["service".into()]).unwrap();
        let config = Config::with_root_certificates(
            host,
            port,
            &[server.cert.der().as_ref()],
            &[client.cert.der().as_ref()],
            &client.key_pair.serialize_der(),
        )
        .unwrap();

        let server_config = ServerConfig::builder()
            .client_auth(ClientAuthMode::Required, client.cert.pem().as_bytes())
            .disable_ktls()
            .with_der_bytes(
                &[server.cert.der().as_ref()],
                &server.key_pair.serialize_der(),
            )
            .unwrap();
        (config, TlsAcceptor::new(server_config))
    }

    async fn ready(transport: &Transport) {
        timeout(Duration::from_secs(5), transport.wait_connected())
            .await
            .unwrap()
            .unwrap();
    }

    async fn backing_off(state: &mut watch::Receiver<ConnectionState>) {
        timeout(Duration::from_secs(5), async {
            loop {
                if matches!(*state.borrow(), ConnectionState::BackingOff { .. }) {
                    return;
                }
                state.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
    }

    async fn request(connection: &mut FramedConn<impl NetworkStream, Netstring>) -> Request {
        let bytes = timeout(Duration::from_secs(5), connection.recv())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        json::from_slice(&bytes).unwrap()
    }

    async fn respond(
        connection: &mut FramedConn<impl NetworkStream, Netstring>,
        request: &Request,
    ) {
        let response = Response::result(request.id().unwrap().clone(), request.method());
        connection
            .send(&json::to_vec(&response).unwrap())
            .await
            .unwrap();
    }

    fn assert_transport_error(error: ClientError, expected: fn(&TransportError) -> bool) {
        let ClientError::Transport(error) = error else {
            panic!("expected transport error, got {error:?}");
        };
        assert!(
            expected(error.downcast_ref::<TransportError>().unwrap()),
            "{error}"
        );
    }

    async fn capacity_test_transport(capacity_timeout: Duration) -> (Transport, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config::plaintext("127.0.0.1", listener.local_addr().unwrap().port())
            .unwrap()
            .with_capacity_timeout(capacity_timeout);
        let transport = Transport::start(config);
        let (peer, _) = timeout(Duration::from_secs(5), listener.accept())
            .await
            .unwrap()
            .unwrap();
        ready(&transport).await;
        (transport, peer)
    }

    #[tokio::test]
    async fn capacity_wait_resumes_when_a_slot_is_released() {
        let (transport, _peer) = capacity_test_transport(DEFAULT_CAPACITY_TIMEOUT).await;
        let permits = transport
            .inner
            .in_flight
            .try_acquire_many(MAX_IN_FLIGHT as u32)
            .unwrap();
        let waiting = transport.session();
        tokio::pin!(waiting);
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("operation did not wait for capacity"),
            _ = std::future::ready(()) => {}
        }
        drop(permits);
        let (_session, permit) = waiting.await.unwrap();
        assert_eq!(
            transport.inner.in_flight.available_permits(),
            MAX_IN_FLIGHT - 1
        );
        drop(permit);
        assert_eq!(transport.inner.in_flight.available_permits(), MAX_IN_FLIGHT);
    }

    #[tokio::test]
    async fn capacity_wait_returns_busy_only_after_the_configured_timeout() {
        let capacity_timeout = Duration::from_millis(100);
        let (transport, _peer) = capacity_test_transport(capacity_timeout).await;
        let _permits = transport
            .inner
            .in_flight
            .try_acquire_many(MAX_IN_FLIGHT as u32)
            .unwrap();
        tokio::time::pause();
        let waiting = transport.session();
        tokio::pin!(waiting);
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("operation did not wait for capacity"),
            _ = std::future::ready(()) => {}
        }
        tokio::time::advance(capacity_timeout - Duration::from_millis(1)).await;
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("operation returned busy before the timeout"),
            _ = std::future::ready(()) => {}
        }
        tokio::time::advance(Duration::from_millis(1)).await;
        assert!(matches!(waiting.await, Err(TransportError::Busy)));
    }

    #[tokio::test]
    async fn capacity_wait_stops_on_shutdown_or_session_loss() {
        for shutdown in [true, false] {
            let (transport, peer) = capacity_test_transport(Duration::from_secs(60)).await;
            let _permits = transport
                .inner
                .in_flight
                .try_acquire_many(MAX_IN_FLIGHT as u32)
                .unwrap();
            let waiting = transport.session();
            tokio::pin!(waiting);
            tokio::select! {
                biased;
                _ = &mut waiting => panic!("operation did not wait for capacity"),
                _ = std::future::ready(()) => {}
            }
            if shutdown {
                transport.shutdown();
            }
            drop(peer);
            let result = timeout(Duration::from_secs(5), waiting).await.unwrap();
            if shutdown {
                assert!(matches!(result, Err(TransportError::Stopped)));
            } else {
                assert!(matches!(result, Err(TransportError::Disconnected)));
            }
        }
    }

    #[tokio::test]
    async fn capacity_wait_cancellation_does_not_consume_a_slot() {
        let (transport, _peer) = capacity_test_transport(Duration::from_secs(60)).await;
        let permits = transport
            .inner
            .in_flight
            .try_acquire_many(MAX_IN_FLIGHT as u32)
            .unwrap();
        {
            let waiting = transport.session();
            tokio::pin!(waiting);
            tokio::select! {
                biased;
                _ = &mut waiting => panic!("operation did not wait for capacity"),
                _ = std::future::ready(()) => {}
            }
        }
        drop(permits);
        assert_eq!(transport.inner.in_flight.available_permits(), MAX_IN_FLIGHT);
        assert!(transport.session().await.is_ok());
        assert_eq!(transport.inner.in_flight.available_permits(), MAX_IN_FLIGHT);
    }

    #[tokio::test]
    async fn capacity_wait_zero_timeout_is_fail_fast() {
        let (transport, _peer) = capacity_test_transport(Duration::ZERO).await;
        assert!(transport.session().await.is_ok());
        let _permits = transport
            .inner
            .in_flight
            .try_acquire_many(MAX_IN_FLIGHT as u32)
            .unwrap();
        tokio::time::pause();
        let started = Instant::now();
        assert!(matches!(
            transport.session().await,
            Err(TransportError::Busy)
        ));
        assert!(started.elapsed() < Duration::from_millis(10));
    }

    #[tokio::test]
    async fn tcp_connections_enable_keepalive_and_nodelay() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let stream = connect_tcp("127.0.0.1", listener.local_addr().unwrap().port())
            .await
            .unwrap();

        assert!(SockRef::from(&stream).keepalive().unwrap());
        assert!(stream.nodelay().unwrap());
    }

    #[test]
    fn plaintext_configuration_needs_no_credentials_and_validates_hosts() {
        let config = Config::plaintext("127.0.0.1", 443).unwrap();
        assert!(config.tls.is_none());
        assert_eq!(config.max_response_frame_size, DEFAULT_MAX_FRAME_SIZE);
        let (tls, _) = tls_config(443);
        assert!(tls.tls.is_some());
        for host in ["", "local\0host", "\u{e9}"] {
            assert!(matches!(
                Config::plaintext(host, 443),
                Err(error) if error.kind() == io::ErrorKind::InvalidInput
            ));
        }
    }

    #[tokio::test]
    async fn plaintext_multiplexes_calls_and_notifications() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config::plaintext("127.0.0.1", listener.local_addr().unwrap().port())
            .unwrap()
            .with_max_response_frame_size(256);
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(socket, Netstring);
            let notification = request(&mut connection).await;
            assert!(notification.is_notification());
            assert_eq!(notification.method(), "notify");
            let first = request(&mut connection).await;
            let second = request(&mut connection).await;
            respond(&mut connection, &second).await;
            respond(&mut connection, &first).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        client.notify("notify", ()).await.unwrap();
        let (first, second) = tokio::join!(
            client.call::<_, String>("first", ()),
            client.call::<_, String>("second", ()),
        );
        assert_eq!(first.unwrap(), "first");
        assert_eq!(second.unwrap(), "second");
        transport.shutdown();
        assert!(matches!(
            transport.wait_connected().await,
            Err(TransportError::Stopped)
        ));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn plaintext_reconnects_without_replaying_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config::plaintext("127.0.0.1", listener.local_addr().unwrap().port()).unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut first = FramedConn::new(socket, Netstring);
            assert_eq!(request(&mut first).await.method(), "old");
            drop(first);
            let (socket, _) = listener.accept().await.unwrap();
            let mut second = FramedConn::new(socket, Netstring);
            let call = request(&mut second).await;
            assert_eq!(call.method(), "new");
            respond(&mut second, &call).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        assert_transport_error(
            client.call::<_, String>("old", ()).await.unwrap_err(),
            |error| matches!(error, TransportError::Session(_)),
        );
        ready(&transport).await;
        assert_eq!(client.call::<_, String>("new", ()).await.unwrap(), "new");
        server.await.unwrap();
    }

    #[test]
    fn rejects_missing_client_credentials_during_configuration() {
        let certificate = rcgen::generate_simple_self_signed(vec!["service".into()]).unwrap();
        let certificates = &[certificate.cert.der().as_ref()];
        let key = certificate.key_pair.serialize_der();
        let missing_certificate = Config::new("localhost", 443, &[], &key);
        let missing_key = Config::new("localhost", 443, certificates, &[]);
        assert!(matches!(
            missing_certificate,
            Err(error) if error.kind() == io::ErrorKind::InvalidInput
        ));
        assert!(matches!(
            missing_key,
            Err(error) if error.kind() == io::ErrorKind::InvalidInput
        ));
    }

    #[test]
    fn config_uses_system_roots_without_supplied_server_certificates() {
        let client = rcgen::generate_simple_self_signed(vec!["service".into()]).unwrap();
        assert!(
            Config::new(
                "localhost",
                443,
                &[client.cert.der().as_ref()],
                &client.key_pair.serialize_der(),
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn system_roots_reject_an_untrusted_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (_, acceptor) = tls_config(port);
        let client = rcgen::generate_simple_self_signed(vec!["service".into()]).unwrap();
        let config = Config::new(
            "localhost",
            port,
            &[client.cert.der().as_ref()],
            &client.key_pair.serialize_der(),
        )
        .unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            assert!(acceptor.accept(socket).await.is_err());
        });
        let error = timeout(Duration::from_secs(5), connect(&config))
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("CERTIFICATE_VERIFY_FAILED"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn verifies_ip_subject_alternative_names() {
        for mismatched in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let certificate_name = if mismatched { "127.0.0.2" } else { "127.0.0.1" };
            let (config, acceptor) = tls_config_for_host(
                listener.local_addr().unwrap().port(),
                "127.0.0.1",
                certificate_name,
            );
            let server = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let result = acceptor.accept(socket).await;
                if mismatched {
                    assert!(result.is_err());
                } else {
                    result.unwrap().shutdown().await.unwrap();
                }
            });
            let result = timeout(Duration::from_secs(5), connect(&config))
                .await
                .unwrap();
            assert_eq!(result.is_err(), mismatched);
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn multiplexes_out_of_order_calls_over_mutually_authenticated_tls() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let stream = acceptor.accept(socket).await.unwrap();
            assert!(stream.peer_certificate_der().unwrap().is_some());
            let mut connection = FramedConn::new(stream, Netstring);
            let first = request(&mut connection).await;
            let second = request(&mut connection).await;
            respond(&mut connection, &second).await;
            respond(&mut connection, &first).await;
        });
        let transport = Transport::start(config);
        let shared = transport.clone();
        drop(transport);
        ready(&shared).await;
        let client = Client::new(shared).with_request_timeout(Duration::from_secs(5));
        let (first, second) = tokio::join!(
            client.call::<_, String>("first", ()),
            client.call::<_, String>("second", ()),
        );
        assert_eq!(first.unwrap(), "first");
        assert_eq!(second.unwrap(), "second");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn notifications_do_not_wait_for_reply_frames() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let notification = request(&mut connection).await;
            assert!(notification.is_notification());
            assert_eq!(notification.method(), "notify");
            let call = request(&mut connection).await;
            respond(&mut connection, &call).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let client = Client::new(transport).with_request_timeout(Duration::from_secs(5));
        client.notify("notify", ()).await.unwrap();
        assert_eq!(client.call::<_, String>("call", ()).await.unwrap(), "call");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn reconnects_after_abrupt_failure_without_replaying_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut first = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            assert_eq!(request(&mut first).await.method(), "old");
            drop(first);
            let (socket, _) = listener.accept().await.unwrap();
            let mut second = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let call = request(&mut second).await;
            assert_eq!(call.method(), "new");
            respond(&mut second, &call).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        assert_transport_error(
            client.call::<_, String>("old", ()).await.unwrap_err(),
            |error| matches!(error, TransportError::Session(_)),
        );
        ready(&transport).await;
        assert_eq!(client.call::<_, String>("new", ()).await.unwrap(), "new");
        server.await.unwrap();
    }

    #[test]
    fn backoff_is_exponential_jittered_and_capped() {
        let mut backoff = Backoff::default();
        for attempt in 0..100 {
            let base = INITIAL_BACKOFF
                .saturating_mul(1_u32 << attempt.min(7))
                .min(MAX_BACKOFF);
            let delay = backoff.next();
            assert!(delay >= base / 2);
            assert!(delay <= (base * 3 / 2).min(MAX_BACKOFF));
        }
        assert_eq!(backoff.attempt, 100);
    }

    #[tokio::test]
    async fn reconnects_on_idle_eof_without_any_rpc_calls() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            for _ in 0..3 {
                let (socket, _) = listener.accept().await.unwrap();
                let mut stream = acceptor.accept(socket).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        let _transport = Transport::start(config);
        timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn reconnects_after_an_initial_connection_refusal() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (config, acceptor) = tls_config(address.port());
        drop(listener);
        let transport = Transport::start(config);
        let mut state = transport.subscribe();
        backing_off(&mut state).await;
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        assert_transport_error(
            client.call::<_, String>("offline", ()).await.unwrap_err(),
            |error| matches!(error, TransportError::Disconnected),
        );
        let listener = TcpListener::bind(address).await.unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let call = request(&mut connection).await;
            assert_eq!(call.method(), "ready");
            respond(&mut connection, &call).await;
        });
        ready(&transport).await;
        assert_eq!(
            client.call::<_, String>("ready", ()).await.unwrap(),
            "ready"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn cancelling_a_response_wait_keeps_the_session_usable() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let (received, received_call) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            assert_eq!(request(&mut connection).await.method(), "cancel");
            received.send(()).unwrap();
            let next = request(&mut connection).await;
            respond(&mut connection, &next).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let client =
            Arc::new(Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5)));
        let pending_client = client.clone();
        let pending =
            tokio::spawn(async move { pending_client.call::<_, String>("cancel", ()).await });
        timeout(Duration::from_secs(5), received_call)
            .await
            .unwrap()
            .unwrap();
        pending.abort();
        assert!(pending.await.unwrap_err().is_cancelled());
        assert_eq!(transport.inner.in_flight.available_permits(), MAX_IN_FLIGHT);
        assert_eq!(client.call::<_, String>("next", ()).await.unwrap(), "next");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn malformed_frames_trigger_reconnection_without_peer_eof() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let (invalidate, invalidation) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut first = acceptor.accept(socket).await.unwrap();
            invalidation.await.unwrap();
            first.write_all(b"1:x!").await.unwrap();
            first.flush().await.unwrap();
            let (socket, _) = listener.accept().await.unwrap();
            let mut second = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let call = request(&mut second).await;
            respond(&mut second, &call).await;
            drop(first);
        });
        let transport = Transport::start(config);
        let mut state = transport.subscribe();
        ready(&transport).await;
        invalidate.send(()).unwrap();
        backing_off(&mut state).await;
        ready(&transport).await;
        let client = Client::new(transport).with_request_timeout(Duration::from_secs(5));
        assert_eq!(
            client.call::<_, String>("fresh", ()).await.unwrap(),
            "fresh"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn enforces_operation_limit_without_sending_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let call = request(&mut connection).await;
            assert_eq!(call.method(), "healthy");
            respond(&mut connection, &call).await;
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let permits = transport
            .inner
            .in_flight
            .acquire_many(MAX_IN_FLIGHT as u32)
            .await
            .unwrap();
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        assert_transport_error(
            client.call::<_, String>("blocked", ()).await.unwrap_err(),
            |error| matches!(error, TransportError::Busy),
        );
        drop(permits);
        assert_eq!(
            client.call::<_, String>("healthy", ()).await.unwrap(),
            "healthy"
        );
        server.await.unwrap();
    }

    #[test]
    fn response_frame_size_defaults_and_overrides() {
        let (config, _) = tls_config(443);
        assert_eq!(config.max_response_frame_size, DEFAULT_MAX_FRAME_SIZE);
        let config = config.with_max_response_frame_size(256);
        assert_eq!(config.max_response_frame_size, 256);
        let config = config.with_max_response_frame_size(0);
        assert_eq!(config.max_response_frame_size, 0);
    }

    #[tokio::test]
    async fn large_outgoing_frames_survive_reconnection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let limit = 20 * 1024 * 1024;
        let payload_size = 17 * 1024 * 1024;
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (socket, _) = listener.accept().await.unwrap();
                let mut connection =
                    FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring)
                        .with_max_frame_size(limit);
                let notification = request(&mut connection).await;
                assert!(notification.is_notification());
                assert_eq!(notification.method(), "notify");
                let (payload,): (String,) = notification.params_as().unwrap();
                assert_eq!(payload.len(), payload_size);
                let call = request(&mut connection).await;
                let (payload,): (String,) = call.params_as().unwrap();
                assert_eq!(payload.len(), payload_size);
                respond(&mut connection, &call).await;
            }
        });
        let transport = Transport::start(config);
        let mut state = transport.subscribe();
        let client = Client::new(transport.clone()).with_request_timeout(Duration::from_secs(5));
        for generation in 0..2 {
            ready(&transport).await;
            client
                .notify("notify", ("x".repeat(payload_size),))
                .await
                .unwrap();
            assert_eq!(
                client
                    .call::<_, String>("upload", ("x".repeat(payload_size),))
                    .await
                    .unwrap(),
                "upload"
            );
            if generation == 0 {
                backing_off(&mut state).await;
            }
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn enforces_response_frame_limit() {
        for limit in [64, 256, 0] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
            let server = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let mut connection =
                    FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
                let call = request(&mut connection).await;
                assert_eq!(call.method(), "short");
                let response = Response::result(call.id().unwrap().clone(), "x".repeat(128));
                connection
                    .send(&json::to_vec(&response).unwrap())
                    .await
                    .unwrap();
            });
            let config = config.with_max_response_frame_size(limit);
            let transport = Transport::start(config);
            ready(&transport).await;
            let client = Client::new(transport).with_request_timeout(Duration::from_secs(5));
            let response = client.call::<_, String>("short", ()).await;
            if limit == 64 {
                assert_transport_error(response.unwrap_err(), |error| {
                    matches!(error, TransportError::Session(_))
                });
            } else {
                assert_eq!(response.unwrap(), "x".repeat(128));
            }
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn response_frame_limit_does_not_limit_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            let call = request(&mut connection).await;
            respond(&mut connection, &call).await;
        });
        let transport = Transport::start(config.with_max_response_frame_size(64));
        ready(&transport).await;
        let client = Client::new(transport).with_request_timeout(Duration::from_secs(5));
        assert_eq!(
            client
                .call::<_, String>("wide", ("x".repeat(256),))
                .await
                .unwrap(),
            "wide"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_untrusted_servers_and_hostname_mismatches() {
        for untrusted in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let certificate_name = if untrusted {
                "localhost"
            } else {
                "wrong.example"
            };
            let (mut config, acceptor) = tls_config_for_host(
                listener.local_addr().unwrap().port(),
                "localhost",
                certificate_name,
            );
            if untrusted {
                let unrelated =
                    rcgen::generate_simple_self_signed(vec!["unrelated".into()]).unwrap();
                config.tls = Some(Arc::new(
                    ClientConfig::builder()
                        .with_root_certs_der_bytes(&[unrelated.cert.der().as_ref()])
                        .disable_ktls()
                        .build()
                        .unwrap(),
                ));
            }
            let server = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                assert!(acceptor.accept(socket).await.is_err());
            });
            let result = timeout(Duration::from_secs(5), connect(&config))
                .await
                .unwrap();
            assert!(result.is_err());
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn tls_handshake_has_a_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, _) = tls_config(listener.local_addr().unwrap().port());
        let attempt = tokio::spawn(async move { connect(&config).await });
        let (_socket, _) = timeout(Duration::from_secs(5), listener.accept())
            .await
            .unwrap()
            .unwrap();
        tokio::time::pause();
        tokio::time::advance(CONNECT_TIMEOUT).await;
        let error = timeout(Duration::from_secs(1), attempt)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn supervisor_preserves_stopped_state_when_session_closes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config::plaintext("127.0.0.1", listener.local_addr().unwrap().port()).unwrap();
        let sessions = Arc::new(Mutex::new(SessionSlot::default()));
        let (state, mut observed) = watch::channel(ConnectionState::Connecting);
        let supervisor = tokio::spawn(supervise(config, sessions.clone(), state.clone()));
        let (peer, _) = timeout(Duration::from_secs(5), listener.accept())
            .await
            .unwrap()
            .unwrap();
        timeout(
            Duration::from_secs(5),
            observed.wait_for(|state| matches!(state, ConnectionState::Connected)),
        )
        .await
        .unwrap()
        .unwrap();
        {
            let mut slot = sessions.lock().unwrap();
            assert!(matches!(*slot, SessionSlot::Occupied(_)));
            *slot = SessionSlot::Stopped;
            state.send_replace(ConnectionState::Stopped);
        }
        drop(peer);
        timeout(Duration::from_secs(5), supervisor)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(*sessions.lock().unwrap(), SessionSlot::Stopped));
        assert!(matches!(*observed.borrow(), ConnectionState::Stopped));
    }

    #[tokio::test]
    async fn shutdown_stops_connection_attempts_and_future_calls() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, _) = tls_config(listener.local_addr().unwrap().port());
        let transport = Transport::start(config);
        let state = transport.subscribe();
        transport.shutdown();
        assert!(matches!(
            transport.wait_connected().await,
            Err(TransportError::Stopped)
        ));
        assert!(matches!(
            transport.round_trip(b"{}".to_vec()).await,
            Err(TransportError::Stopped)
        ));
        assert!(matches!(*state.borrow(), ConnectionState::Stopped));
        assert!(matches!(
            *transport.inner.session.lock().unwrap(),
            SessionSlot::Stopped
        ));
    }

    #[tokio::test]
    async fn shutdown_cancels_pending_calls_across_clones() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let (received, received_call) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = FramedConn::new(acceptor.accept(socket).await.unwrap(), Netstring);
            request(&mut connection).await;
            received.send(()).unwrap();
            let closed = timeout(Duration::from_secs(5), connection.recv())
                .await
                .unwrap();
            assert!(matches!(closed, Ok(None) | Err(_)));
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let clone = transport.clone();
        let client = Client::new(transport).with_request_timeout(Duration::from_secs(5));
        let pending = tokio::spawn(async move { client.call::<_, String>("pending", ()).await });
        timeout(Duration::from_secs(5), received_call)
            .await
            .unwrap()
            .unwrap();
        clone.shutdown();
        let error = timeout(Duration::from_secs(5), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_transport_error(error, |error| matches!(error, TransportError::Stopped));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn dropping_the_last_clone_closes_the_idle_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (config, acceptor) = tls_config(listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut stream = acceptor.accept(socket).await.unwrap();
            let mut buffer = [0];
            let closed = timeout(Duration::from_secs(5), stream.read(&mut buffer))
                .await
                .unwrap();
            assert!(matches!(closed, Ok(0) | Err(_)));
        });
        let transport = Transport::start(config);
        ready(&transport).await;
        let state = transport.subscribe();
        let clone = transport.clone();
        drop(transport);
        assert!(matches!(*state.borrow(), ConnectionState::Connected));
        drop(clone);
        assert!(matches!(*state.borrow(), ConnectionState::Stopped));
        server.await.unwrap();
    }
}
