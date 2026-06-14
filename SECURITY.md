# Security Policy

## Credential Handling

DeltaForge stores all credentials in a `.env` file that is excluded from version control via `.gitignore`. Credentials are never logged, committed, or exposed through the dashboard API.

**Never commit:**
- `.env` files containing real credentials
- API keys, tokens, or passwords in source code
- Database files containing trade data

## Dashboard Security

The dashboard binds to `127.0.0.1` (localhost only) by default. If you need remote access:

1. Set `DASHBOARD_API_TOKEN` in `.env` to require authentication for mutating endpoints.
2. Set `DASHBOARD_CORS_ORIGINS` to restrict allowed origins.
3. Use a reverse proxy (e.g., nginx) with TLS for production deployments.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainers with a description of the vulnerability.
3. Include steps to reproduce if possible.
4. Allow reasonable time for a fix before public disclosure.

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| < 2.0   | No        |
