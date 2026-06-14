# Dashboard

DeltaForge includes a real-time web dashboard built with FastAPI.

## Starting the Dashboard

```bash
make dashboard            # standalone at http://localhost:8900
make run                  # trading + dashboard together
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | KPI overview (capital, P&L, risk state) |
| `/api/capital` | GET | Current capital state |
| `/api/trades` | GET | Trade history (filterable) |
| `/api/trades/summary` | GET | Daily summary |
| `/api/risk` | GET | Risk gate status |
| `/api/engine` | GET | Live engine state |
| `/api/halt` | POST | Toggle kill switch |
| `/ws/live` | WS | Real-time updates via WebSocket |

## WebSocket

Connect to `/ws/live` for real-time updates. The server pushes JSON messages for:

- Trade entries and exits
- Capital changes
- Risk gate state changes
- Engine status updates

## Security

The dashboard is configured for security by default:

- **Bind address**: `127.0.0.1` (localhost only). Change via `DASHBOARD_HOST` in `.env`.
- **Port**: `8900`. Change via `DASHBOARD_PORT`.
- **CORS**: Restricted to localhost origins. Configure via `DASHBOARD_CORS_ORIGINS`.
- **Authentication**: Set `DASHBOARD_API_TOKEN` in `.env` to require a Bearer token for the `/api/halt` endpoint.

If you need remote access, bind to `0.0.0.0` and **always** set `DASHBOARD_API_TOKEN` to protect the halt endpoint.

## Alerts Configuration

The dashboard works alongside the alert system. Configure alerts in `.env`:

| Method | Variables |
|--------|-----------|
| Slack (recommended) | `ALERT_METHOD=slack`, `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` |
| iMessage | `ALERT_METHOD=imessage`, `IMESSAGE_RECIPIENT` (macOS only) |
| Telegram | `ALERT_METHOD=telegram`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
