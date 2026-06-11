# Deployment: cloud face + office backend

This preview layout keeps the public surface small:

- `face` runs on the VPS, serves `index.html`, terminates HTTPS, and reverse-proxies API paths.
- `backend` runs in the office, owns docs, datastore, prompts, app DB, and LM Studio access.
- A private tunnel connects the VPS to the office backend.

## Office backend

Create a stable session secret:

```bash
export LUXRIOT_AUTH_SECRET="$(openssl rand -hex 32)"
```

Start the backend container on the office server:

```bash
docker compose -f docker-compose.office.yml up -d --build
```

By default, the container expects LM Studio on the office host:

```bash
LMSTUDIO_BASE_URL=http://host.docker.internal:1234
```

If LM Studio runs elsewhere on the office network, set `LMSTUDIO_BASE_URL` to that URL.

The office container mounts:

- `./docs` as `/data/docs` read-only
- `./datastore` as `/data/datastore`
- `./backend/data` as `/data` for `app.sqlite`

## Tunnel

For a quick preview, use a reverse SSH tunnel from the office server to the VPS:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -R 127.0.0.1:18080:127.0.0.1:8000 \
  tunnel-user@assistant.example.com
```

The sample systemd service lives at `deploy/tunnel/luxriot-reverse-tunnel.service`.
Replace the placeholders before installing it.

## Cloud face with real HTTPS

On the VPS, point DNS to the server and open ports `80` and `443`.
Caddy will obtain and renew certificates automatically:

```bash
export SITE_ADDRESS=assistant.example.com
export BACKEND_UPSTREAM=http://host.docker.internal:18080
docker compose -f docker-compose.cloud.yml up -d --build
```

Caddy forwards backend requests with `Host: localhost`, so the backend can keep a narrow
`LUXRIOT_TRUSTED_HOSTS=localhost,127.0.0.1` setting behind the tunnel.

Add the customer IP allowlist at the VPS firewall or hosting firewall layer.

## Cloud face with self-signed HTTPS

For an IP-only or lab preview, use Caddy internal TLS:

```bash
export SITE_ADDRESS=:443
export BACKEND_UPSTREAM=http://host.docker.internal:18080
docker compose -f docker-compose.cloud.yml -f docker-compose.cloud.selfsigned.yml up -d --build
```

Browsers will warn unless the Caddy internal CA is trusted on the client machine.
Use real DNS + automatic HTTPS for partner-facing demos whenever possible.

## Backend security env

Recommended same-origin HTTPS values for the office backend:

```bash
export LUXRIOT_AUTH_SECRET='replace-with-long-random-value'
export LUXRIOT_TRUSTED_HOSTS='assistant.example.com,localhost,127.0.0.1'
export LUXRIOT_COOKIE_SECURE=1
export LUXRIOT_COOKIE_SAMESITE=lax
export LUXRIOT_RAWDOCS_REQUIRE_AUTH=1
export LUXRIOT_ASSETS_REQUIRE_AUTH=1
```

For split-origin deployments, also set explicit CORS origins and use `SameSite=None; Secure`.
