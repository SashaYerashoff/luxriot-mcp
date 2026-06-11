# Luxriot assistant RTX backend handoff

Дата состояния: 2026-06-11.

Этот файл нужен, чтобы продолжить деплой на другой машине/в другом Codex-сеансе без потери контекста.

## Что уже сделано

- Git branch: `feature/cloud-office-gateway-prep`.
- VPS: `178.104.125.49`, Ubuntu 26.04, hostname `llm-ubuntu-8gb-nbg1-1`.
- DNS: `llm.luxriot.systems -> 178.104.125.49`.
- На VPS установлен Docker и поднят cloud face container из `docker-compose.cloud.yml`.
- HTTPS/Let's Encrypt работает. Caddy терминирует TLS на VPS и проксирует backend API через tunnel.
- Текущий VPS `/opt/luxriot-mcp/.env`:

```bash
SITE_ADDRESS=llm.luxriot.systems
BACKEND_UPSTREAM=http://host.docker.internal:18080
```

- Проверено на VPS:

```text
curl -I http://llm.luxriot.systems/     -> HTTP/1.1 308 Permanent Redirect
curl -I https://llm.luxriot.systems/    -> HTTP/2 200
```

## Цель следующего этапа

На RTX/backend машине поднять backend container, убедиться что:

1. FastAPI отвечает на `/health`.
2. Backend видит локальный datastore/index.
3. Backend container видит LM Studio `/v1/models`.
4. Reverse SSH tunnel соединяет VPS Caddy с backend.
5. Снаружи `http://llm.luxriot.systems/health` отдает backend health через tunnel.

## Важное про данные

Git branch содержит код и Docker-конфиги, но не содержит рабочие данные:

- `docs/`
- `datastore/`
- `backend/data/app.sqlite`

Перед запуском backend эти директории/файлы должны быть на RTX машине рядом с `docker-compose.office.yml`.

Проверка в корне репозитория на RTX:

```bash
test -d docs && du -sh docs
test -d datastore && du -sh datastore
test -f backend/data/app.sqlite && ls -lh backend/data/app.sqlite
test -f datastore/evo_1_32/index.sqlite && ls -lh datastore/evo_1_32/index.sqlite
test -f datastore/evo_1_32/pages.jsonl && wc -l datastore/evo_1_32/pages.jsonl
```

Если этих данных нет, их нужно скопировать/rsync-нуть отдельно с рабочей машины или с текущего источника данных.

## Clone/update branch на RTX

```bash
mkdir -p /opt
cd /opt
git clone -b feature/cloud-office-gateway-prep https://github.com/SashaYerashoff/luxriot-mcp.git
cd luxriot-mcp
```

Если репозиторий уже есть:

```bash
cd /opt/luxriot-mcp
git fetch origin
git checkout feature/cloud-office-gateway-prep
git pull --ff-only
```

## Docker на RTX

Если Docker отсутствует:

```bash
apt update
apt install -y ca-certificates curl git docker.io docker-compose-v2
systemctl enable --now docker
docker --version
docker compose version
```

## Backend `.env` на RTX

В корне `/opt/luxriot-mcp`:

```bash
cat > .env <<'EOF'
LUXRIOT_AUTH_SECRET=replace-with-output-of-openssl-rand-hex-32
LUXRIOT_ADMIN_USERNAME=admin
LUXRIOT_ADMIN_PASSWORD=replace-with-strong-admin-password
LMSTUDIO_BASE_URL=http://host.docker.internal:1234
LUXRIOT_TRUSTED_HOSTS=localhost,127.0.0.1
LUXRIOT_COOKIE_SECURE=1
LUXRIOT_COOKIE_SAMESITE=lax
LUXRIOT_RAWDOCS_REQUIRE_AUTH=1
LUXRIOT_ASSETS_REQUIRE_AUTH=1
EOF
```

Создать секрет можно так:

```bash
openssl rand -hex 32
```

Для HTTPS deploy важно оставить:

```bash
LUXRIOT_COOKIE_SECURE=1
```

Если временно откатываемся на чистый HTTP без TLS, поменять на `0`.

## Запуск backend

```bash
cd /opt/luxriot-mcp
docker compose -f docker-compose.office.yml up -d --build
docker compose -f docker-compose.office.yml ps
docker compose -f docker-compose.office.yml logs --tail=120 backend
```

Если нужно сбросить пароль admin без удаления БД, временно добавить в `.env`:

```bash
LUXRIOT_ADMIN_USERNAME=admin
LUXRIOT_ADMIN_PASSWORD=replace-with-new-strong-password
LUXRIOT_ADMIN_PASSWORD_RESET=1
```

Затем перезапустить backend:

```bash
docker compose -f docker-compose.office.yml up -d --force-recreate
docker compose -f docker-compose.office.yml logs --tail=80 backend
```

После успешного входа удалить или очистить reset flag:

```bash
sed -i '/^LUXRIOT_ADMIN_PASSWORD_RESET=/d' .env
docker compose -f docker-compose.office.yml up -d --force-recreate
```

Важно: `docker-compose.office.yml` должен пробрасывать `LUXRIOT_ADMIN_USERNAME`,
`LUXRIOT_ADMIN_PASSWORD` и `LUXRIOT_ADMIN_PASSWORD_RESET` в container environment.

Проверка health локально на RTX:

```bash
curl -H 'Host: localhost' http://127.0.0.1:8000/health
```

Ожидаемо:

```json
{
  "status": "ok",
  "datastore_ready": true,
  "embeddings_ready": true
}
```

`/health` не проверяет LM Studio. LM Studio проверяется отдельно.

## Проверка LM Studio из backend container

```bash
docker compose -f docker-compose.office.yml exec -T backend python - <<'PY'
import os, httpx
url = os.environ["LMSTUDIO_BASE_URL"].rstrip("/") + "/v1/models"
print("GET", url)
r = httpx.get(url, timeout=10)
print(r.status_code)
print(r.text[:3000])
PY
```

Если это не работает:

- проверить, запущен ли LM Studio server;
- проверить, что LM Studio слушает не только `127.0.0.1`, а доступен контейнеру через `host.docker.internal`;
- в LM Studio включить bind/listen на `0.0.0.0` или другой доступный interface;
- если LM Studio на другом IP, поменять `LMSTUDIO_BASE_URL`, например `http://192.168.0.X:1234`.

## Проверка поиска по индексу без LM Studio

```bash
curl -sS -H 'Host: localhost' \
  -H 'Content-Type: application/json' \
  -d '{"query":"installation prerequisites","k":3}' \
  http://127.0.0.1:8000/docs/search | head -c 2000
echo
```

Если `datastore_ready=true`, но search не работает, смотреть:

```bash
docker compose -f docker-compose.office.yml logs --tail=200 backend
```

## Reverse SSH tunnel RTX -> VPS

Временный ручной запуск с RTX:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18080:127.0.0.1:8000 \
  root@178.104.125.49
```

Этот процесс держать открытым в отдельном терминале.

После tunnel проверить с RTX или любой машины:

```bash
curl https://llm.luxriot.systems/health
curl -sS -H 'Content-Type: application/json' \
  -d '{"query":"installation prerequisites","k":3}' \
  https://llm.luxriot.systems/docs/search | head -c 2000
echo
```

Если tunnel поднят, но `llm.luxriot.systems/health` не работает, проверить на VPS:

```bash
curl http://127.0.0.1:18080/health
docker compose -f /opt/luxriot-mcp/docker-compose.cloud.yml logs --tail=120 face
```

Если `curl http://127.0.0.1:18080/health` на VPS работает, но Caddy container
все равно отдает `502`, значит tunnel слушает только loopback VPS, а Caddy
container не может подключиться к нему напрямую. В таком случае на VPS нужен
локальный bridge через `socat` на gateway docker-сети:

```bash
cd /opt/luxriot-mcp

GATEWAY=$(docker network inspect luxriot-mcp_default --format '{{(index .IPAM.Config 0).Gateway}}')
SUBNET=$(docker network inspect luxriot-mcp_default --format '{{(index .IPAM.Config 0).Subnet}}')
echo "gateway=$GATEWAY subnet=$SUBNET"

ufw allow from "$SUBNET" to "$GATEWAY" port 18080 proto tcp

pgrep -fa 'socat.*18080' || \
  nohup socat TCP-LISTEN:18080,bind="$GATEWAY",fork,reuseaddr TCP:127.0.0.1:18080 \
    >/var/log/luxriot-socat-18080.log 2>&1 &

cat > .env <<EOF
SITE_ADDRESS=llm.luxriot.systems
BACKEND_UPSTREAM=http://$GATEWAY:18080
EOF

docker compose -f docker-compose.cloud.yml up -d --force-recreate

docker compose -f docker-compose.cloud.yml exec -T face sh -lc \
  "wget -S -O- -T 5 http://$GATEWAY:18080/health"
curl -sS https://llm.luxriot.systems/health
```

## Минимальные порты для админа

Текущий HTTPS preview:

```text
Internet/browser -> VPS 178.104.125.49:443/tcp
Internet/ACME    -> VPS 178.104.125.49:80/tcp и 443/tcp для Let's Encrypt renewals
RTX/backend      -> VPS 178.104.125.49:22/tcp для reverse SSH tunnel
VPS localhost    -> 127.0.0.1:18080, только локально, наружу не открывать
RTX local        -> 127.0.0.1:8000 для backend container publish
backend container -> LM Studio :1234
```

Если временно откатываемся на HTTP-only:

```text
Internet/browser -> VPS 178.104.125.49:80/tcp
```

Важно: на VPS UFW уже настроен так, что `22/tcp` открыт только для текущего dev IP `78.84.119.234`. Для tunnel с RTX нужен public egress IP той сети, где находится RTX. Узнать на RTX:

```bash
curl ifconfig.me
```

Потом на VPS добавить:

```bash
ufw allow from <RTX_PUBLIC_IP> to any port 22 proto tcp
ufw status verbose
```

## Prompt для нового Codex-сеанса на RTX

Можно начать новый сеанс так:

```text
Прочитай deploy/RUNBOOK_RTX_BACKEND.md и продолжи деплой с раздела "Цель следующего этапа".
Контекст: VPS face уже поднят на https://llm.luxriot.systems/ с Let's Encrypt сертификатом, backend еще не поднят.
Нужно поднять docker-compose.office.yml на RTX, проверить /health, datastore, LM Studio /v1/models, затем поднять reverse SSH tunnel на VPS 178.104.125.49:18080.
HTTPS на VPS уже работает; backend наружу не открывать, использовать reverse SSH tunnel.
```
