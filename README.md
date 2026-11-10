# OmniQEC

Project page: <https://grahamyan.github.io/OmniQEC/>

## Repository layout

```text
webpage/   Project homepage, deployed to GitHub Pages by .github/workflows/deploy.yml
code/      Source code
data/      Data
```

## Webpage

The webpage includes the paper homepage, processed manuscript figures, a
downloadable manuscript preview, and a paper-grounded AI reading assistant with
an offline mock mode. Every push to `main` redeploys `webpage/` to GitHub Pages.

### Local preview

Open one terminal at the repository root:

```bash
cd webpage
python3 -m http.server 8000 --bind 127.0.0.1
```

Open a second terminal:

```bash
cd webpage/paper-assistant-worker
npm install
npm run dev
```

Then open <http://127.0.0.1:8000/>.

The default assistant is deterministic and fully local. It performs keyword and
term-weighted retrieval over curated OmniQEC manuscript excerpts and never calls
a model.

### Validate

From `webpage/paper-assistant-worker/`:

```bash
npm run typecheck
npm test
npm run dry-run
```

`npm run dry-run` builds the reserved Workers AI configuration without
deploying it. The production endpoint and GitHub Pages origin should remain
unset until the page is approved.
