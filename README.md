# OmniQEC research preview

This is a local-only static preview for the OmniQEC manuscript. It includes the
paper homepage, processed manuscript figures, a downloadable manuscript
preview, and a paper-grounded AI reading assistant with an offline mock mode.

No Git remote, commit, push, GitHub Pages setting, Workers AI connection, or
Cloudflare deployment is created by this project.

## Start the local preview

Open one terminal at the project root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open a second terminal:

```bash
cd paper-assistant-worker
npm install
npm run dev
```

Then open <http://127.0.0.1:8000/>.

The default assistant is deterministic and fully local. It performs keyword and
term-weighted retrieval over curated OmniQEC manuscript excerpts and never calls
a model.

## Validate

From `paper-assistant-worker/`:

```bash
npm run typecheck
npm test
npm run dry-run
```

`npm run dry-run` builds the reserved Workers AI configuration without
deploying it. The production endpoint and GitHub Pages origin should remain
unset until the page is approved.
