# Newzire (`english-news`)

Automated English digital news portal covering **India & World** affairs, built with a fast zero-CDN static site generator (`news_bot.py`) and scheduled GitHub Actions publishing.

## Architecture
- **Static Site Builder & AI News Bot (`news_bot.py`)**: Scans multi-beat RSS feeds (India, World, Politics, Economy, Technology, Sports), ranks candidates with India-First editorial weights, rewrites original 350–450 word English journalism with 100% Fact Lock via a 15-model Google Gemini → NVIDIA NIM failover chain, and builds static SEO pages.
- **Standalone Production CSS (`style.css`)**: Warm editorial dark (`#0e0e0e`) and light (`#f2efe9`) modes with zero render-blocking CSS frameworks.
- **CMS Schema Compatibility**: Uses the same `site_config.json`, `articles_db.json`, `pages/*.html`, `uploads/`, and `.github/workflows/schedule.yml` (`mode: rebuild`) structure as `auto-news-bot` / `aabtak24-admin`.

## Local Usage

```bash
pip install -r requirements.txt

# Rebuild all static HTML, sitemap.xml, and robots.txt from articles_db.json
SKIP_IMAGE_CHECK=1 python3 news_bot.py --rebuild-only

# Run unit tests
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## GitHub Actions Setup
1. Add `GEMINI_API_KEY` (and optionally `NVIDIA_API_KEY`) under **Settings → Secrets and variables → Actions**.
2. Enable GitHub Pages under **Settings → Pages → Source: Deploy from a branch (`main` / root)**.
3. When ready to attach a custom domain, add a `CNAME` file and update `"site_url"` in `site_config.json`.
