# OmniQEC paper assistant

This directory contains the paper-grounded assistant used by the local OmniQEC
preview page. The default development command is fully local: it performs the
same deterministic lexical retrieval as the production Worker but does not log
in to Cloudflare or call a model.

## Local mock

```bash
npm install
npm run types
npm test
npm run dev
```

The service listens on `http://127.0.0.1:8787`:

- `GET /health`
- `POST /chat` with `{ "message": "...", "history": [...] }`

Only `http://localhost:8000` and `http://127.0.0.1:8000` are accepted browser
origins by the mock configuration.

## Reserved production configuration

`wrangler.jsonc` declares an `AI` binding and keeps
`@cf/zai-org/glm-4.7-flash` as the configured model. It is intentionally not
deployed or connected during local review. Both Wrangler configurations allow
only the two local `:8000` origins for now. Add the final GitHub Pages origin and
update the front-end `data-production-endpoint` only after the page is approved.

No model credential is stored in the front end or this directory.
