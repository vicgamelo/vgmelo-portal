# vgmelo-portal

Auth, quotas and demo gateway for [vgmelo.tech](https://vgmelo.tech). The static site calls it on the
same origin under `/api/*`; it puts the live demos (agent orchestrator, genai-lens traces, credit
scoring, MCP tools) behind login, per-user daily quotas and a global daily token budget.

FastAPI, SQLAlchemy 2 (Postgres in production, SQLite in tests), Authlib, argon2, pyotp, httpx2.

## API

All responses are JSON unless noted. Every non-GET request must carry `Origin: https://vgmelo.tech`
(otherwise `403 {"detail":"bad_origin"}`). Bodies are capped at `MAX_BODY_BYTES`.

| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| GET | `/api/me` | | `{user: {id, name, email, provider, mfa} \| null, pending: "totp_setup" \| null, providers: {google, github, email}, quotas: {agents, score, mcp, traces, studio: {limit, used, remaining}}, tokens: {limit, used, remaining}, resets_at, budget_exhausted}` |
| GET | `/api/auth/{google\|github}/start?next=/path` | | 302 to the provider. `next` must be a relative path, anything else becomes `/`. |
| GET | `/api/auth/{google\|github}/callback` | | 302 to `SITE_URL + next`; on failure adds `?auth_error=oauth`. |
| POST | `/api/auth/email/register` | `{email, password}` | Always `202 {"status":"verification_sent"}`. Password 10-128 chars. |
| POST | `/api/auth/email/verify` | `{email, code}` | Same browser that registered. `{"status":"totp_setup_required"}` and a setup-only session. |
| POST | `/api/auth/totp/setup` | | `{otpauth_uri, qr_svg}` (needs the setup session). |
| POST | `/api/auth/totp/confirm` | `{code}` | `{"status":"ok", "recovery_codes": [10 codes, shown once]}`, full session. |
| POST | `/api/auth/email/login` | `{email, password, totp \| recovery_code}` | `{"status":"ok"}`, or `{"status":"totp_setup_required"}` if setup was never finished. Any wrong factor: `401 {"detail":"invalid_credentials"}`. |
| POST | `/api/auth/logout` | | 204, session deleted server side. |
| POST | `/api/demo/agents/stream` | `{question, servers?}` | `text/event-stream` relayed from the orchestrator; the last event is `done` with `run_id`, `trace_id`, tokens. `servers` picks the MCP servers on for the run (`brasil`, `noticias`, `normas`, `agenda`); omitted means all. |
| GET | `/api/demo/agents/servers` | | Which MCP servers are up and which agents each one feeds (for the demo's switches). |
| GET | `/api/studio/catalog` | | Agent studio: models, tools, starting templates and limits (from the orchestrator). |
| GET | `/api/studio/versions` | | The visitor's saved agents, newest version first. Private to each user. |
| POST | `/api/studio/versions` | `{name, note?, agent}` | Saves the next version of `name`. `agent` is `{instructions, model, tools, temperature, max_steps}`. Free, capped by `STUDIO_MAX_VERSIONS`. |
| DELETE | `/api/studio/versions/{id}` | | Deletes one of your versions (404 for anyone else's). |
| POST | `/api/studio/run/stream` | `{question, agent}` | Runs the agent as written, same SSE as the agents demo. Spends the `studio` quota; `422 {reason: invalid_agent, message}` when the orchestrator refuses the model or a tool. |
| GET | `/api/demo/traces/{trace_id}` | | genai-lens spans; 404 unless the trace came from this user's run (also 404 while the lens hasn't received it yet). |
| POST | `/api/demo/score` | one applicant (see `mlops-platform/examples/applicants.json`) | `{model_version, prediction: {default_probability, decision}}` |
| POST | `/api/demo/score/pipeline` | `{applicant: {18 fields: text, number or null}, lang: pt\|en}` | `text/event-stream` from the scoring service: `stage` events (`raw`, `transform`, `normalize`, `fit`, `model`, `explain`), then `result` and `done`. An unusable profile is `422 {"detail":{"reason":"invalid_applicant","problems":[{column, code, message, value, severity}]}}`, refunded. |
| POST | `/api/demo/score/batch/validate?lang=` | the CSV as the raw body, `content-type: text/csv` (max `BATCH_MAX_BYTES`) | The validation report (by line and column). Free, `BATCH_VALIDATE_PER_HOUR` per user. |
| POST | `/api/demo/score/batch?lang=` | same | `{model_version, report, results, csv}`; one use of the `score` quota per file, refunded with `422 {"detail":{"reason":"no_valid_rows","report"}}` when nothing could be scored. |
| POST | `/api/demo/score/agent` | `{field, answer (≤120 chars)}` | The credit demo's micro-agent: `CREDIT_AGENT_MODEL` on Groq maps one free-text answer to `{status: ok, value}`, `{status: ask, options}` (ambiguous), `dont_know` or `unknown`. Only values in the credit-g domain come out. Tokens count toward `GLOBAL_DAILY_TOKENS`; `CREDIT_AGENT_PER_HOUR` per user; doesn't use the `score` quota. `503 agent_off` without `GROQ_API_KEY`. |
| POST | `/api/demo/mcp/{tool}` | `{args}` | Tools: `consultar_cep`, `consultar_cnpj`, `validar_documento`, `feriados_nacionais`, `dias_uteis`, `indicador_economico`, `listar_municipios`, `consultar_ddd`. `{tool, result}`; tool errors are `422 {"detail":{"reason":"tool_error","message"}}`. |
| POST | `/api/contact` | `{name, email, kind: job\|consulting\|other, message, website, started_at}` | Always `{"status":"sent"}`. `website` is a honeypot, `started_at` is epoch ms when the form rendered. |
| GET | `/admin` | | HTML for `ADMIN_EMAILS` (verified email, full session); 404 for everyone else. |
| GET | `/api/health` | | |

Demos need a full session (`401 {"detail":"login_required"}`). Limits answer `429` with
`{"reason": ..., "retry_at": ISO-8601}` and `Retry-After`:

- `daily_quota`: this user's quota for that demo is used up until midnight in `QUOTA_TZ`.
- `global_budget`: the agents demo, the studio and the credit micro-agent spent `GLOBAL_DAILY_TOKENS` today, for everyone. The owner gets an
  e-mail at 80% and 100% of it (`ALERT_BUDGET_PERCENTS`, once per day each).
- `user_tokens`: this user spent `USER_DAILY_TOKENS` today across agents, studio and the credit micro-agent. A run
  starts only with balance left, so the last one of the day may end a little over.
- `busy`: an agents run of yours is still going, or `MAX_CONCURRENT_RUNS` are in flight.
- `rate_limited` / `locked`: too many requests or failed logins from this IP / for this account.

Upstream failures are `502 {"detail":"upstream_unavailable"}` and give the quota unit back.

## Configuration

Everything comes from the environment; see [`.env.example`](.env.example). Required: `DATABASE_URL`,
`SECRET_KEY` (32+ chars, used to HMAC short codes and encrypt TOTP secrets). Google/GitHub login turns on
when both client id and secret are set; email login needs `RESEND_API_KEY`.

## Run

```bash
uv sync
uv run pytest -q                       # SQLite
PORTAL_TEST_DATABASE_URL=postgresql+psycopg://u:p@localhost:5432/db uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy
docker build -t vgmelo-portal .
```

The image runs as uid 10001, works with `read_only: true`, and must run a single process (the agents
concurrency limit is in memory). In the `vgmelo.tech` site block of Caddy, `TRUSTED_PROXIES` set to Caddy's
Docker network:

```
@portal path /api/* /admin
handle @portal {
	reverse_proxy vgmelo-portal:8000 {
		flush_interval -1  # SSE
	}
}
```

## Threat model

- **Abuse and cost.** Every demo needs a login; quotas per user/day/demo, a global token budget fed by the
  orchestrator's `done` event (and topped up from the lens spans when they add up to more), one run per user,
  a cap on concurrent runs, question and body size limits. A run is counted even if the visitor disconnects.
- **Accounts.** argon2id passwords, mandatory TOTP for email accounts (±30 s window, each step accepted once),
  hashed recovery codes, verification codes bound to the registering browser, lockout per account and per IP.
  Errors never say whether an email exists. OAuth uses `state` bound to a cookie and PKCE (S256); accounts are
  never merged by email across providers.
- **Sessions.** Random token in a `__Host-` cookie (`HttpOnly; Secure; SameSite=Lax`), only its SHA-256 in the
  database, rotated on every login, deleted on logout, absolute expiry.
- **Web.** Origin check on every non-GET request, relative-only `next`, escaped admin page with a strict CSP,
  `no-store` on the API, client IP from `X-Forwarded-For` only when the peer is a trusted proxy.
- **Secrets.** Internal keys stay server side; logs carry method, path and status only (no query strings,
  bodies, tokens or codes).
- **Out of scope.** One person can hold one account per provider (three quotas); a determined attacker can
  lock a known email account by failing logins on purpose (the lock is temporary).
