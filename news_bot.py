import os
import re
import sys
import copy
import html
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote, urlparse, parse_qs
import requests
import feedparser
from bs4 import BeautifulSoup

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

GEMINI_HTTP_TIMEOUT_MS = 45_000
NIM_HTTP_TIMEOUT = (10, 45)
IMAGE_CHECK_TIMEOUT_SEC = 6


def build_google_client(key):
    """Creates a Google GenAI client with fail-fast HTTP timeouts when supported."""
    if not key or genai is None:
        return None
    if genai_types is not None and hasattr(genai_types, "HttpOptions"):
        http_kwargs = {"timeout": GEMINI_HTTP_TIMEOUT_MS}
        if hasattr(genai_types, "HttpRetryOptions"):
            try:
                http_kwargs["retry_options"] = genai_types.HttpRetryOptions(attempts=1)
            except Exception:
                pass
        try:
            return genai.Client(
                api_key=key,
                http_options=genai_types.HttpOptions(**http_kwargs),
            )
        except Exception:
            pass
    return genai.Client(api_key=key)


# 1. Load AI Provider Secret Keys
api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
client = build_google_client(api_key) if api_key else None

# 2. Brand, Site, Multi-Beat RSS Feeds & Editorial Settings
DEFAULT_SITE_URL = "https://adityacoder1.github.io/english-news"
SITE_URL = DEFAULT_SITE_URL
SITE_NAME = "Newzire"
ARCHIVE_FILE = "articles_db.json"
NEWS_DIR = "news"
CONFIG_FILE = "site_config.json"
GENERATED_MANIFEST = ".generated_pages.json"
DEFAULT_FALLBACK_IMAGE = (
    "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80"
)

# A. SOURCE BIAS: Multi-beat India & World RSS feeds
RSS_FEEDS = [
    {"url": "https://www.thehindu.com/news/national/feeder/default.rss", "source": "The Hindu", "weight": 9.0},
    {"url": "https://indianexpress.com/section/india/feed/", "source": "Indian Express", "weight": 8.5},
    {"url": "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml", "source": "BBC News", "weight": 8.0},
    {"url": "https://feeds.feedburner.com/ndtvnews-top-stories", "source": "NDTV", "weight": 7.5},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "source": "BBC World", "weight": 8.5},
    {"url": "https://www.thehindu.com/news/international/feeder/default.rss", "source": "The Hindu World", "weight": 8.0},
    {"url": "https://www.thehindu.com/business/feeder/default.rss", "source": "The Hindu Business", "weight": 8.0},
    {"url": "https://indianexpress.com/section/technology/feed/", "source": "Indian Express Tech", "weight": 8.0},
    {"url": "https://feeds.bbci.co.uk/sport/rss.xml", "source": "BBC Sport", "weight": 6.5},
]

# B. KEYWORD BIAS: Boosts high-impact national, global, tech, and economic reporting
KEYWORD_BIAS = {
    "india": 3.5,
    "ai": 5.0,
    "artificial intelligence": 5.0,
    "isro": 4.5,
    "semiconductor": 4.0,
    "economy": 3.5,
    "gdp": 3.5,
    "rbi": 3.0,
    "trade": 3.0,
    "imec": 3.5,
    "quad": 3.5,
    "parliament": 3.0,
    "supreme court": 3.0,
    "policy": 2.5,
    "geopolitics": 3.0,
    "summit": 2.5,
    "defense": 2.5,
    "defence": 2.5,
    "cricket": 1.5,
    "startup": 3.0,
}

BLOCKED_KEYWORDS = [
    "horoscope", "astrology", "wordle", "crossword", "sponsored", "betting", "coupon", "casino"
]

# C. AI SELECTION BIAS: Which story should the editor select from the top 5 candidates?
SELECTION_BIAS_PROMPT = """
- Prioritize high-impact stories covering India's economic progress, deep-tech/AI/ISRO milestones, national security, major governance or judicial reforms, global diplomacy, world affairs, and citizen-relevant developments.
- Include compelling World, Business, Technology, and major Sports milestones when they carry national or global significance.
- Skip routine petty crime, celebrity gossip, clickbait listicles, or hyper-local trivia.
"""

# D. AI FRAMING & PHRASING BIAS: Sharp India-First English Journalism + 100% Fact Lock
TONE_AND_FRAMING_BIAS = """
- Editorial Lens: Present the story from an informed, forward-looking, 'India-First' progressive analytical perspective that highlights strategic significance, technological capability, economic opportunity, and real impact on ordinary citizens.
- Voice & Style: Write crisp, authoritative, publication-grade English journalism (The Economist / Financial Times clarity meets modern digital newsroom pace). Open with a sharp lead paragraph, use informative <h2> subheadings, and close with forward-looking context.
- Strict 100% Fact Lock: Never alter, embellish, or fabricate any fact, statistic, proper name, date, quote, or verified event from the source material. Apply the editorial voice purely through structure, framing, word choice, and headline emphasis.
"""

# 3. Fallback Site Configuration (100% CMS-compatible with aabtak24-admin)
DEFAULT_SITE_CONFIG = {
    "site_url": DEFAULT_SITE_URL,
    "categories": [
        {"id": "india", "name": "India", "file": "india.html", "active_for_ai": True, "aliases": ["India", "National", "Nation", "Domestic"]},
        {"id": "world", "name": "World", "file": "world.html", "active_for_ai": True, "aliases": ["World", "International", "Global", "Diplomacy", "Foreign Affairs"]},
        {"id": "politics", "name": "Politics", "file": "politics.html", "active_for_ai": True, "aliases": ["Politics", "Policy", "Parliament", "Governance", "Elections"]},
        {"id": "economy", "name": "Economy", "file": "economy.html", "active_for_ai": True, "aliases": ["Economy", "Business", "Markets", "Finance", "Trade"]},
        {"id": "technology", "name": "Technology", "file": "technology.html", "active_for_ai": True, "aliases": ["Technology", "Tech", "Science", "AI", "Space"]},
        {"id": "sports", "name": "Sports", "file": "sports.html", "active_for_ai": True, "aliases": ["Sports", "Cricket", "Athletics"]},
        {"id": "opinion", "name": "Opinion", "file": "opinion.html", "active_for_ai": True, "aliases": ["Opinion", "Analysis", "Editorial", "Explainers", "Features", "Interviews"]},
        {"id": "culture", "name": "Culture", "file": "culture.html", "active_for_ai": True, "aliases": ["Culture", "Entertainment", "Lifestyle", "Film", "Books", "Travel", "Wellness"]},
    ],
    "custom_pages": [
        {
            "id": "about",
            "title": "About Us | Newzire",
            "meta_description": "Learn about Newzire — our independent English digital newsroom covering India & World affairs.",
            "source_fragment": "pages/about.html",
            "output_file": "about.html",
            "include_in_sitemap": True,
            "changefreq": "monthly",
        },
        {
            "id": "contact",
            "title": "Contact Us | Newzire",
            "meta_description": "Contact the Newzire editorial team for story tips, corrections, or partnerships.",
            "source_fragment": "pages/contact.html",
            "output_file": "contact.html",
            "include_in_sitemap": True,
            "changefreq": "monthly",
        },
        {
            "id": "editorial-policy",
            "title": "Editorial Policy | Newzire",
            "meta_description": "Read Newzire's editorial standards, 100% Fact Lock sourcing principles, and corrections policy.",
            "source_fragment": "pages/editorial-policy.html",
            "output_file": "editorial-policy.html",
            "include_in_sitemap": True,
            "changefreq": "yearly",
        },
        {
            "id": "privacy-policy",
            "title": "Privacy Policy | Newzire",
            "meta_description": "Read the Newzire Privacy Policy covering browser preferences, logs, and cookies.",
            "source_fragment": "pages/privacy-policy.html",
            "output_file": "privacy-policy.html",
            "include_in_sitemap": True,
            "changefreq": "yearly",
        },
    ],
    "header": {
        "enabled": True,
        "theme": "dark",
        "layout": "compact",
        "sticky": True,
        "show_logo": True,
        "brand_title": "Newzire",
        "brand_tagline": "India & World Digital Edition",
        "nav_style": "underline",
        "top_bar_enabled": True,
        "top_bar_show_date": True,
        "top_bar_text": "New Delhi • Global Desk",
        "edition_tag": "DIGITAL EDITION • INDIA & WORLD",
        "hamburger_style": "bars",
        "mobile_menu_style": "slide-right",
        "mobile_theme": "auto",
        "mobile_menu_label": "Menu",
        "mobile_show_tagline": True,
        "theme_toggle_enabled": True,
        "default_color_mode": "dark",
        "action_links": [],
        "colors": {},
    },
    "header_menu": [
        {"label": "Home", "href": "index.html"},
        {"label": "India", "href": "india.html"},
        {"label": "World", "href": "world.html"},
        {"label": "Politics", "href": "politics.html"},
        {"label": "Economy", "href": "economy.html"},
        {"label": "Technology", "href": "technology.html"},
        {"label": "Sports", "href": "sports.html"},
        {"label": "Opinion", "href": "opinion.html"},
        {"label": "Culture", "href": "culture.html"},
    ],
    "footer": {
        "enabled": True,
        "theme": "dark",
        "show_logo": True,
        "brand_title": "Newzire",
        "brand_tagline": "Fast, fact-locked English journalism across India and the world. Sharp reporting on politics, geopolitics, economy, technology, sports, and culture.",
        "contact_label": "Editorial Desk:",
        "contact_email": "contact@newzire.com",
        "columns": [
            {
                "id": "news",
                "title": "News & Beats",
                "links": [
                    {"label": "India (National)", "href": "india.html"},
                    {"label": "World (Global)", "href": "world.html"},
                    {"label": "Politics & Policy", "href": "politics.html"},
                    {"label": "Economy & Markets", "href": "economy.html"},
                    {"label": "Technology & AI", "href": "technology.html"},
                    {"label": "Sports", "href": "sports.html"},
                ],
            },
            {
                "id": "views",
                "title": "Views & Culture",
                "links": [
                    {"label": "Opinion & Analysis", "href": "opinion.html"},
                    {"label": "Culture & Lifestyle", "href": "culture.html"},
                    {"label": "Markets & Explainers", "href": "economy.html"},
                    {"label": "Science & Space", "href": "technology.html"},
                    {"label": "Visual Stories", "href": "index.html#visual-markets-section"},
                    {"label": "Reader Services", "href": "index.html#magazine-services-section"},
                ],
            },
            {
                "id": "pages",
                "title": "Newzire & Policies",
                "links": [
                    {"label": "About Newzire", "href": "about.html"},
                    {"label": "Editorial Policy", "href": "editorial-policy.html"},
                    {"label": "Privacy Policy", "href": "privacy-policy.html"},
                    {"label": "Contact & News Tips", "href": "contact.html"},
                ],
            },
        ],
        "social_links": [
            {"label": "X", "href": "https://x.com/newzire_in", "new_tab": True},
            {"label": "Facebook", "href": "https://facebook.com/NewzireNews", "new_tab": True},
            {"label": "Instagram", "href": "https://instagram.com/newzire.news", "new_tab": True},
        ],
        "copyright_text": "© {year} Newzire Digital Media. All Rights Reserved.",
        "bottom_note": "DIGITAL EDITION • INDIA & WORLD",
        "colors": {},
    },
    "promo_banner": {
        "enabled": True,
        "badge": "SPONSORED • DEAL OF THE DAY",
        "highlight": "India Tech & Reader Pass • Flat 30% Off",
        "headline": "Annual Productivity & Cloud Learning Bundle — Exclusive 30% Reader Discount on Career, AI & Finance Certifications",
        "cta_label": "View Offer →",
        "modal_title": "India Tech & Reader Pass — Exclusive 30% Off",
        "modal_summary": "Newzire readers receive an instant 30% partner discount on accredited AI engineering, financial markets, and digital journalism micro-courses across India.",
        "promo_code": "NEWZIRE30",
        "footnote": "* Limited-time partner offer valid for Newzire digital edition readers.",
    },
    "homepage": {
        "preset": "newzire-editorial",
        "deduplicate_across_sections": True,
        "sections": [
            {
                "id": "editorial-frontpage",
                "enabled": True,
                "title": "Frontpage Desk",
                "subtitle": "Editor's Picks • Lead Story • Latest & Trending",
                "category": "all",
                "layout": "editorial-desk",
                "max_articles": 11,
                "show_view_all": False,
                "bg_theme": "default",
            },
            {
                "id": "cat-india",
                "enabled": True,
                "title": "India",
                "subtitle": "National Desk",
                "category": "india",
                "layout": "grid",
                "max_articles": 4,
                "show_view_all": True,
                "bg_theme": "default",
            },
            {
                "id": "cat-world",
                "enabled": True,
                "title": "World",
                "subtitle": "Global & Diplomacy",
                "category": "world",
                "layout": "grid",
                "max_articles": 4,
                "show_view_all": True,
                "bg_theme": "default",
            },
            {
                "id": "cat-politics",
                "enabled": True,
                "title": "Politics & Policy",
                "subtitle": "Governance & Parliament",
                "category": "politics",
                "layout": "grid",
                "max_articles": 4,
                "show_view_all": True,
                "bg_theme": "default",
            },
            {
                "id": "cat-economy",
                "enabled": True,
                "title": "Tech & Economy",
                "subtitle": "Markets, AI & Innovation",
                "category": "economy",
                "layout": "grid",
                "max_articles": 4,
                "show_view_all": True,
                "bg_theme": "default",
            },
        ],
    },
}

HOMEPAGE_LAYOUTS = ("editorial-desk", "bento", "grid", "magazine", "compact-list", "strip")
SECTION_BG_THEMES = ("auto", "default", "sand", "crimson", "sage", "slate", "amber")
VALID_COLOR_MODES = ("light", "dark", "system")

HOMEPAGE_PRESETS = {
    "newzire-editorial": {
        "name": "Newzire 5-Band Editorial",
        "deduplicate_across_sections": True,
        "sections": copy.deepcopy(DEFAULT_SITE_CONFIG["homepage"]["sections"]),
    },
    "wire-editorial": {
        "name": "Wire Editorial Desk",
        "deduplicate_across_sections": True,
        "sections": copy.deepcopy(DEFAULT_SITE_CONFIG["homepage"]["sections"]),
    },
    "classic-bento": {
        "name": "Classic Bento",
        "deduplicate_across_sections": True,
        "sections": [
            {
                "id": "top-stories",
                "enabled": True,
                "title": "Top Stories",
                "subtitle": "Latest Updates",
                "category": "all",
                "layout": "bento",
                "max_articles": 5,
                "show_view_all": False,
                "bg_theme": "auto",
            },
            {
                "id": "more-news",
                "enabled": True,
                "title": "More News",
                "subtitle": "All Desks",
                "category": "all",
                "layout": "grid",
                "max_articles": 12,
                "show_view_all": False,
                "bg_theme": "auto",
            },
        ],
    },
}

# 4. Multi-Provider AI Model Fallback Configuration (15-Model Chain)
NVIDIA_NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL_FALLBACK_CHAIN = [
    # Tier 1: Google Gemini (GEMINI_API_KEY)
    {"provider": "google", "model": "gemini-3.8-flash", "wait_sec": 2},
    {"provider": "google", "model": "gemini-4-hydrogen", "wait_sec": 2},
    {"provider": "google", "model": "gemini-3.6-flash", "wait_sec": 2},
    {"provider": "google", "model": "gemini-3.5-flash-lite", "wait_sec": 2},
    {"provider": "google", "model": "gemini-3.1-flash-lite", "wait_sec": 2},
    {"provider": "google", "model": "gemini-3-flash-preview", "wait_sec": 2},
    # Tier 2: Google Open-Weight Gemma 4 (GEMINI_API_KEY)
    {"provider": "google", "model": "gemma-4-31b-it", "wait_sec": 2},
    {"provider": "google", "model": "gemma-4-26b-a4b-it", "wait_sec": 2},
    # Tier 3: Open-Source Frontier Models via NVIDIA NIM (NVIDIA_API_KEY)
    {"provider": "nvidia_nim", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "mistralai/mistral-large-4-1t-a52b", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "nvidia/nemotron-3-ultra-550b-a55b", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "mistralai/mistral-small-4-119b-2603", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "qwen/qwen2.5-72b-instruct", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "openai/gpt-oss-120b", "wait_sec": 2},
    {"provider": "nvidia_nim", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "wait_sec": 2},
]

MODELS_TO_TRY = [entry["model"] for entry in MODEL_FALLBACK_CHAIN]

# 5. National & Global Reader Services Desks (Band 5 + Navbar Services Dropdown)
READER_SERVICE_DESKS = {
    "health-wellness": {
        "badge": "Advice Desk • Health",
        "title": "Health & Wellness — Medical & Preventive Care",
        "summary": "Evidence-backed guidance on seasonal immunity, preventive screenings, mental well-being, and accessing national cashless emergency networks across India.",
        "highlights": [
            "PM-RAHAT & Ayushman Bharat: How government medical colleges process cashless trauma and emergency admissions.",
            "Preventive Checkups: Recommended annual cardiac, metabolic, and vitamin panels for working professionals.",
            "Send a health desk query anytime to health@newzire.com for our Sunday Q&A roundup.",
        ],
        "cta": "Submit a Health Query",
    },
    "legal-consumer": {
        "badge": "Advice Desk • Law & Rights",
        "title": "Legal & Consumer Guide — Know Your Rights",
        "summary": "Clear, plain-English explainers on digital banking fraud reporting, consumer e-commerce protection, tenancy contracts, and Supreme Court landmark rulings.",
        "highlights": [
            "Cyber Fraud Golden Window: Report unauthorized UPI or banking transfers immediately on National Helpline 1930 or cybercrime.gov.in.",
            "E-Commerce & Aviation Refunds: Key consumer commission rules on defective deliveries and flight cancellations.",
            "Panel Insights: Weekly explainers reviewed with senior Supreme Court and High Court practitioners.",
        ],
        "cta": "Ask the Legal Desk",
    },
    "career-exams": {
        "badge": "Utility Desk • Education",
        "title": "Career & Exams — National Competitive Hub",
        "summary": "Verified exam timelines, syllabus breakdowns, and current-affairs digests for UPSC, SSC, JEE, NEET, CAT, GATE, and international scholarship tracks.",
        "highlights": [
            "UPSC & State PCS: Weekly India & World geopolitical briefing sheets curated from Newzire reports.",
            "National Scholarship Portal (NSP): Track renewal deadlines and institutional verification windows.",
            "AI & Deep-Tech Upskilling: Free accredited NPTEL and IIT-backed micro-certifications for 2026–27.",
        ],
        "cta": "Browse Study Briefings",
    },
    "jobs-desk": {
        "badge": "Utility Desk • Careers",
        "title": "Jobs Desk — Public Sector & Tech Hiring",
        "summary": "Curated national hiring updates spanning central government services, PSU engineering roles, global capability centres, and high-growth AI startups.",
        "highlights": [
            "Public Sector & Railways: Verified notifications for central administrative, banking, and engineering cadres.",
            "Bengaluru, Hyderabad & NCR Tech Hiring: Rising demand for AI systems engineers, data analysts, and chip design talent.",
            "Workforce 2030: How hybrid policies and skills-first credentials are reshaping graduate hiring.",
        ],
        "cta": "Subscribe to Career Alerts",
    },
    "events-calendar": {
        "badge": "Utility Desk • National & Global",
        "title": "Events Calendar — Summits, Tech & Culture",
        "summary": "Key dates for major parliamentary sessions, international diplomatic summits, tech conferences, literary festivals, and sporting fixtures.",
        "highlights": [
            "Space & Science: ISRO SSLV operational launch window and upcoming deep-space mission briefings.",
            "Global Economy & Diplomacy: Bilateral trade delegations, Indo-Pacific forums, and RBI monetary policy reviews.",
            "Arts & Sports: National film festival circuits, book fairs, and international cricket home series schedules.",
        ],
        "cta": "Suggest an Event Listing",
    },
    "civic-directory": {
        "badge": "Utility Desk • Essential Helplines",
        "title": "Civic Directory — All-India Emergency Numbers",
        "summary": "Quick-reference verified national helplines for emergency response, cybercrime reporting, railway assistance, and citizen grievance portals.",
        "highlights": [
            "All-in-One National Emergency: 112 • Ambulance: 108 • Fire: 101 • Police: 100",
            "National Cyber Crime Helpline: 1930 • National Consumer Helpline: 1915",
            "Indian Railways Inquiry & RailMadad: 139 • Women's Helpline: 181 • Senior Citizens: 14567",
        ],
        "cta": "Bookmark Helplines",
    },
    "fact-check": {
        "badge": "Editorial Integrity • Verification",
        "title": "Fact-Check Desk — 100% Fact Lock Verification",
        "summary": "Our verification desk tracks viral claims, doctored visuals, and misleading statistics across India and global social platforms with transparent primary sources.",
        "highlights": [
            "Source Verification: Every Newzire report cross-checks official government releases, court filings, and Tier-1 wire dispatches.",
            "Spotted a viral claim or factual error? Send the link and claim to factcheck@newzire.com.",
            "Corrections Log: All verified updates are reflected directly on the published story page.",
        ],
        "cta": "Submit a Claim to Verify",
    },
    "archives": {
        "badge": "Digital Archive • Newzire",
        "title": "News Archives — Explore Every Desk",
        "summary": "Browse Newzire's full indexed archive organized by national politics, global geopolitics, economy, technology, sports, opinion, and culture.",
        "highlights": [
            "Every Newzire article has a permanent SEO URL under /news/<slug>.html and is indexed in sitemap.xml.",
            "Use the Quick Desk bar or category pages to filter full coverage across all eight newsroom desks.",
            "RSS and full editorial policy references remain publicly documented for readers and researchers.",
        ],
        "cta": "Open National Desk",
    },
}


# ==========================================================================
# CONFIGURATION & NORMALIZATION HELPERS
# ==========================================================================

def get_site_url(config=None):
    """Returns canonical base site URL without trailing slash."""
    if isinstance(config, dict):
        raw = str(config.get("site_url", "") or "").strip()
        if raw.startswith(("http://", "https://")):
            return raw.rstrip("/")
    return SITE_URL.rstrip("/")


def normalize_homepage_config(raw_homepage):
    """Validates and normalizes the homepage layout configuration with safe defaults."""
    default_hp = copy.deepcopy(DEFAULT_SITE_CONFIG["homepage"])
    if not isinstance(raw_homepage, dict):
        return default_hp

    preset = str(raw_homepage.get("preset", "newzire-editorial") or "newzire-editorial").strip().lower()
    if preset not in HOMEPAGE_PRESETS and preset != "custom":
        preset = "custom"

    dedupe = bool(raw_homepage.get("deduplicate_across_sections", True))

    raw_sections = raw_homepage.get("sections")
    if not isinstance(raw_sections, list):
        preset_data = HOMEPAGE_PRESETS.get(preset) or HOMEPAGE_PRESETS["newzire-editorial"]
        raw_sections = copy.deepcopy(preset_data["sections"])

    normalized_sections = []
    for idx, sec in enumerate(raw_sections):
        if not isinstance(sec, dict):
            continue
        sec_id = str(sec.get("id", "") or "").strip() or f"section-{idx + 1}"
        enabled = bool(sec.get("enabled", True))
        title = str(sec.get("title", "") or "").strip() or f"News Section {idx + 1}"
        subtitle = str(sec.get("subtitle", "") or "").strip()
        category = str(sec.get("category", "all") or "all").strip() or "all"
        layout = str(sec.get("layout", "grid") or "grid").strip().lower()
        if layout not in HOMEPAGE_LAYOUTS:
            layout = "grid"

        default_limit = 11 if layout == "editorial-desk" else (5 if layout == "bento" else 6)
        raw_limit = sec.get("max_articles", default_limit)
        try:
            max_articles = int(raw_limit)
        except (TypeError, ValueError):
            max_articles = default_limit
        max_articles = max(1, min(50, max_articles))

        show_view_all = bool(sec.get("show_view_all", category.lower() != "all"))
        bg_theme = str(sec.get("bg_theme", "auto") or "auto").strip().lower()
        if bg_theme not in SECTION_BG_THEMES:
            bg_theme = "auto"

        normalized_sections.append({
            "id": sec_id,
            "enabled": enabled,
            "title": title,
            "subtitle": subtitle,
            "category": category,
            "layout": layout,
            "max_articles": max_articles,
            "show_view_all": show_view_all,
            "bg_theme": bg_theme,
        })

    if not normalized_sections:
        normalized_sections = copy.deepcopy(default_hp["sections"])

    return {
        "preset": preset,
        "deduplicate_across_sections": dedupe,
        "sections": normalized_sections,
    }


def load_site_config():
    """Loads site_config.json with safe fallback and defaults backfill."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "categories" in data and "header_menu" in data:
                data.setdefault("site_url", DEFAULT_SITE_URL)
                data.setdefault("custom_pages", [])
                header_cfg = data.get("header")
                if not isinstance(header_cfg, dict):
                    data["header"] = copy.deepcopy(DEFAULT_SITE_CONFIG["header"])
                else:
                    for key, default_val in DEFAULT_SITE_CONFIG["header"].items():
                        if key not in header_cfg:
                            header_cfg[key] = copy.deepcopy(default_val)
                    if str(header_cfg.get("default_color_mode", "dark")).strip().lower() not in VALID_COLOR_MODES:
                        header_cfg["default_color_mode"] = "dark"
                footer_cfg = data.get("footer")
                if not isinstance(footer_cfg, dict):
                    data["footer"] = copy.deepcopy(DEFAULT_SITE_CONFIG["footer"])
                else:
                    for key, default_val in DEFAULT_SITE_CONFIG["footer"].items():
                        if key not in footer_cfg:
                            footer_cfg[key] = copy.deepcopy(default_val)
                promo_cfg = data.get("promo_banner")
                if not isinstance(promo_cfg, dict):
                    data["promo_banner"] = copy.deepcopy(DEFAULT_SITE_CONFIG["promo_banner"])
                else:
                    for key, default_val in DEFAULT_SITE_CONFIG["promo_banner"].items():
                        if key not in promo_cfg:
                            promo_cfg[key] = copy.deepcopy(default_val)
                data["homepage"] = normalize_homepage_config(data.get("homepage"))
                if "editors_vault" in data and not isinstance(data.get("editors_vault"), list):
                    data["editors_vault"] = []
                if "admin_lock" in data and data.get("admin_lock") is not None and not isinstance(data.get("admin_lock"), dict):
                    data["admin_lock"] = None
                return data
            print(f"⚠️ {CONFIG_FILE} missing required keys. Using fallback config.")
        except Exception as e:
            print(f"⚠️ Error reading {CONFIG_FILE} ({e}). Using fallback config.")
    else:
        print(f"ℹ️ {CONFIG_FILE} not found. Using built-in default config.")
    return copy.deepcopy(DEFAULT_SITE_CONFIG)


def normalize_ai_category(raw_category, config, ai_categories):
    """Ensures the AI-assigned category always resolves to a valid active category name."""
    cleaned = str(raw_category or "").strip()
    cleaned_lower = cleaned.lower()
    for name in ai_categories:
        if name.lower() == cleaned_lower:
            return name

    for cat in config.get("categories", []):
        if not cat.get("active_for_ai", True):
            continue
        aliases = {str(a).strip().lower() for a in cat.get("aliases", []) if str(a).strip()}
        if cleaned_lower in aliases or cleaned_lower == str(cat.get("id", "")).strip().lower():
            return cat["name"].strip()

    return ai_categories[0]


def clean_slug(slug_text):
    """Ensures the SEO slug only contains clean URL-safe lowercase characters."""
    slug = re.sub(r"[^a-zA-Z0-9\-]", "", str(slug_text or "").lower().replace(" ", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or f"news-{int(time.time())}"


# ==========================================================================
# IMAGE OPTIMIZATION & LOOP-SAFE FALLBACK
# ==========================================================================

IMAGE_FALLBACK_ONERROR = (
    "if(!this.dataset.triedFallback&&this.dataset.fallbackSrc){"
    "this.dataset.triedFallback='1';this.src=this.dataset.fallbackSrc;"
    "}else{this.onerror=null;}"
)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _is_wsrv_url(url):
    try:
        netloc = urlparse(str(url or "").strip()).netloc.lower().split(":")[0]
        return netloc == "wsrv.nl" or netloc.endswith(".wsrv.nl")
    except Exception:
        return False


def normalize_external_url(url):
    cleaned = str(url or "").strip()
    if not cleaned:
        return cleaned
    return quote(cleaned, safe=":/?#[]@!$&'()*+,;=%")


def _extract_raw_from_wsrv(wsrv_url):
    try:
        qs = parse_qs(urlparse(wsrv_url).query)
        origin = (qs.get("url") or [""])[0].strip()
        if origin.startswith(("http://", "https://")):
            return normalize_external_url(origin)
    except Exception:
        pass
    return ""


def optimize_img_url(url, width=800):
    """Compresses & converts external images into fast WebP via wsrv.nl CDN."""
    if not url:
        return url
    cleaned = str(url).strip()
    if not cleaned or not cleaned.startswith(("http://", "https://")) or _is_wsrv_url(cleaned):
        return cleaned
    normalized = normalize_external_url(cleaned)
    return f"https://wsrv.nl/?url={quote(normalized, safe='')}&w={width}&output=webp&q=80"


def should_skip_image_check(explicit=None):
    if explicit is not None:
        return bool(explicit)
    if "--skip-image-check" in sys.argv:
        return True
    return os.environ.get("SKIP_IMAGE_CHECK", "").strip().lower() in {"1", "true", "yes"}


def check_wsrv_image_health(raw_url, width=800):
    """Probes wsrv.nl for a candidate image URL without downloading the body."""
    probe_url = str(raw_url or "").strip()
    if not _is_wsrv_url(probe_url):
        probe_url = optimize_img_url(probe_url, width=width)
    resp = None
    try:
        resp = requests.get(
            probe_url,
            stream=True,
            timeout=IMAGE_CHECK_TIMEOUT_SEC,
            allow_redirects=True,
            headers={"User-Agent": "NewzireNewsBot/1.0"},
        )
        status = getattr(resp, "status_code", 0)
        if 200 <= status < 300:
            raw_headers = getattr(resp, "headers", None) or {}
            raw_ctype = ""
            for k, v in raw_headers.items():
                if str(k).lower() == "content-type":
                    raw_ctype = str(v or "")
                    break
            ctype = raw_ctype.split(";")[0].strip().lower()
            if ctype and not (ctype.startswith("image/") or ctype == "application/octet-stream"):
                return "failed"
            return "healthy"
        if status >= 400:
            return "failed"
        return "transport_error"
    except (requests.exceptions.RequestException, TimeoutError, ConnectionError, OSError):
        return "transport_error"
    finally:
        if resp is not None and hasattr(resp, "close"):
            try:
                resp.close()
            except Exception:
                pass


class ImageResolver:
    """Caches wsrv.nl health checks and falls back gracefully on 404 or transport errors."""

    MAX_CONSECUTIVE_TRANSPORT_ERRORS = 3

    def __init__(self, skip_check=False):
        self.skip_check = bool(skip_check)
        self.disabled = bool(skip_check)
        self._cache = {}
        self._lock = threading.Lock()
        self._transport_error_streak = 0
        self._logged_failures = set()

    def _probe_and_cache(self, cache_key, check_target=None, width=800):
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            if self.disabled:
                return "unknown"

        status = check_wsrv_image_health(check_target or cache_key, width=width)

        with self._lock:
            if status in ("healthy", "failed"):
                self._transport_error_streak = 0
                self._cache[cache_key] = status
                if status == "failed" and cache_key not in self._logged_failures:
                    self._logged_failures.add(cache_key)
                    print(f"⚠️ wsrv.nl failed for image — using direct original URL: {cache_key}")
                return status

            self._transport_error_streak += 1
            if self._transport_error_streak >= self.MAX_CONSECUTIVE_TRANSPORT_ERRORS and not self.disabled:
                self.disabled = True
                print(
                    "⚠️ Skipping remaining build-time wsrv.nl checks after repeated transport errors "
                    "(runtime browser onerror fallback remains active)."
                )
            self._cache[cache_key] = "unknown"
            return "unknown"

    def prewarm(self, raw_urls, max_workers=4):
        if self.skip_check or self.disabled:
            return
        targets = []
        seen = set()
        for url in raw_urls:
            cleaned = str(url or "").strip()
            if not cleaned.startswith(("http://", "https://")):
                continue
            if _is_wsrv_url(cleaned):
                cache_key = _extract_raw_from_wsrv(cleaned) or cleaned
                check_target = cleaned
            else:
                cache_key = normalize_external_url(cleaned)
                check_target = cache_key
            if cache_key not in seen and cache_key not in self._cache:
                seen.add(cache_key)
                targets.append((cache_key, check_target))

        if not targets:
            return

        workers = min(max_workers, len(targets))
        if workers <= 1:
            for cache_key, check_target in targets:
                if self.disabled:
                    break
                self._probe_and_cache(cache_key, check_target=check_target, width=800)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._probe_and_cache, cache_key, check_target, 800)
                for cache_key, check_target in targets
            ]
            for fut in futures:
                try:
                    fut.result()
                except Exception:
                    pass

    def resolve(self, raw_url, width=800, local_prefix=""):
        cleaned = str(raw_url or "").strip()
        if not cleaned:
            cleaned = DEFAULT_FALLBACK_IMAGE

        if not cleaned.startswith(("http://", "https://")):
            if local_prefix and not cleaned.startswith(("../", "/")):
                return f"{local_prefix}{cleaned.lstrip('./')}", ""
            return cleaned, ""

        if _is_wsrv_url(cleaned):
            raw_origin = _extract_raw_from_wsrv(cleaned)
            cache_key = raw_origin or cleaned
            status = self._probe_and_cache(cache_key, check_target=cleaned, width=width)
            if status == "failed" and raw_origin:
                return raw_origin, ""
            return cleaned, (raw_origin if raw_origin and raw_origin != cleaned else "")

        normalized_raw = normalize_external_url(cleaned)
        wsrv_url = optimize_img_url(normalized_raw, width=width)
        status = self._probe_and_cache(normalized_raw, check_target=normalized_raw, width=width)
        if status == "failed":
            return normalized_raw, ""
        return wsrv_url, normalized_raw


def resolve_image_src(raw_url, width=800, local_prefix="", resolver=None):
    active_resolver = resolver or ImageResolver(skip_check=should_skip_image_check())
    return active_resolver.resolve(raw_url, width=width, local_prefix=local_prefix)


def build_img_tag(src, alt, width, height, fallback_src="", extra_attrs="", css_class="editorial-thumb"):
    attrs = [
        f'src="{html.escape(str(src or ""), quote=True)}"',
        f'alt="{html.escape(str(alt or ""), quote=True)}"',
        f'width="{width}"',
        f'height="{height}"',
    ]
    if css_class:
        attrs.append(f'class="{html.escape(css_class, quote=True)}"')
    if fallback_src and fallback_src != src:
        attrs.append(f'data-fallback-src="{html.escape(str(fallback_src), quote=True)}"')
        attrs.append(f'onerror="{IMAGE_FALLBACK_ONERROR}"')
    if extra_attrs and extra_attrs.strip():
        attrs.append(extra_attrs.strip())
    return f"<img {' '.join(attrs)}>"


# ==========================================================================
# AI PARSING, ENGLISH VALIDATION & 15-MODEL FAILOVER
# ==========================================================================

def parse_ai_json(raw_text):
    """Safely parses JSON from Gemini or open-source models even with think tags, fences, or preamble."""
    if not raw_text or not str(raw_text).strip():
        raise ValueError("Empty AI response text")

    cleaned = str(raw_text).strip()
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        fenced_candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced_candidate, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidates = []
    n = len(cleaned)
    i = 0
    while i < n:
        if cleaned[i] == "{":
            depth = 0
            in_string = False
            escape = False
            start = i
            j = i
            while j < n:
                ch = cleaned[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            snippet = cleaned[start : j + 1]
                            try:
                                obj = json.loads(snippet, strict=False)
                                if isinstance(obj, dict):
                                    candidates.append(obj)
                                    i = j
                            except json.JSONDecodeError:
                                pass
                            break
                j += 1
        i += 1

    for obj in candidates:
        if "title" in obj or "content_html" in obj:
            return obj
    if candidates:
        return candidates[0]

    return json.loads(cleaned, strict=False)


def validate_article_payload(article_data):
    """Validates parsed AI output has required article fields in clean English prose."""
    if not isinstance(article_data, dict):
        raise ValueError("AI response JSON must be an object")

    title = article_data.get("title")
    content_html = article_data.get("content_html")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("AI response missing non-empty 'title'")
    if DEVANAGARI_RE.search(title):
        raise ValueError("AI response 'title' must be in English (no Devanagari script)")
    if len(LATIN_WORD_RE.findall(title)) < 2:
        raise ValueError("AI response 'title' lacks sufficient English words")

    if not isinstance(content_html, str) or not content_html.strip():
        raise ValueError("AI response missing non-empty 'content_html'")
    if DEVANAGARI_RE.search(content_html):
        raise ValueError("AI response 'content_html' must be in English (no Devanagari script)")

    plain_text = re.sub(r"<[^>]+>", " ", content_html)
    plain_text = re.sub(r"\s+", " ", plain_text).strip()
    if len(plain_text) < 40 or len(LATIN_WORD_RE.findall(plain_text)) < 10:
        raise ValueError("AI response 'content_html' lacks sufficient English prose")

    raw_idx = article_data.get("selected_index", 0)
    if isinstance(raw_idx, bool):
        article_data["selected_index"] = 0
    elif isinstance(raw_idx, str) and raw_idx.strip().isdigit():
        article_data["selected_index"] = int(raw_idx.strip())

    return article_data


class _ProviderHttpError(RuntimeError):
    """HTTP error carrying status code and credential-failure classification."""

    def __init__(self, message, status_code=None, is_auth=False):
        super().__init__(message)
        self.status_code = status_code
        self.is_provider_auth_error = is_auth


def _extract_http_status(exc):
    for attr in ("code", "status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val < 600:
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int) and 100 <= val < 600:
            return val
    return None


def _is_provider_auth_error(exc, status_code=None):
    if getattr(exc, "is_provider_auth_error", False):
        return True
    code = status_code if status_code is not None else _extract_http_status(exc)
    if code == 401:
        return True
    msg = str(exc or "").lower()
    auth_markers = (
        "api_key_invalid",
        "api key not valid",
        "invalid api key",
        "invalid_api_key",
        "authentication failed",
        "invalid authentication",
        "token expired",
        "key revoked",
        "unauthenticated",
        "unauthorized",
    )
    return any(marker in msg for marker in auth_markers)


def _resolve_google_client():
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        if client is not None and env_key == api_key:
            return client
        return build_google_client(env_key)
    if client is not None and not api_key:
        return client
    if client is not None and api_key and "GEMINI_API_KEY" not in os.environ:
        return None
    return client


def _resolve_nvidia_key():
    if "NVIDIA_API_KEY" in os.environ:
        return (os.environ.get("NVIDIA_API_KEY") or "").strip() or None
    return (nvidia_api_key or "").strip() or None


def _call_google_model(model_name, prompt, google_client):
    try:
        response = google_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        if _is_provider_auth_error(e):
            raise
        status = _extract_http_status(e)
        err_lower = str(e).lower()
        if status == 400 or "400" in err_lower or "response_mime_type" in err_lower or "json" in err_lower or "invalid_argument" in err_lower:
            print(f"ℹ️ {model_name} rejected strict JSON config; retrying without response_mime_type...")
            response = google_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        else:
            raise

    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(f"{model_name} returned empty response text")
    return raw_text


def _call_nvidia_nim_model(model_name, prompt, nim_key):
    headers = {
        "Authorization": f"Bearer {nim_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    base_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior English news editor for Newzire covering India and the world. "
                    "Respond ONLY with a valid raw JSON object matching the requested schema, "
                    "with no markdown fences, thinking tags, or commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 4096,
    }

    resp = requests.post(
        NVIDIA_NIM_API_URL,
        headers=headers,
        json={**base_payload, "response_format": {"type": "json_object"}},
        timeout=NIM_HTTP_TIMEOUT,
    )

    if resp.status_code == 400:
        err_preview = resp.text[:300]
        if _is_provider_auth_error(RuntimeError(err_preview), status_code=400):
            raise _ProviderHttpError(f"HTTP 400: {err_preview}", status_code=400, is_auth=True)
        print(f"ℹ️ {model_name} rejected response_format=json_object (HTTP 400); retrying without response_format...")
        resp = requests.post(
            NVIDIA_NIM_API_URL,
            headers=headers,
            json=base_payload,
            timeout=NIM_HTTP_TIMEOUT,
        )

    if resp.status_code >= 400:
        err_preview = resp.text[:300]
        raise _ProviderHttpError(
            f"HTTP {resp.status_code}: {err_preview}",
            status_code=resp.status_code,
            is_auth=_is_provider_auth_error(RuntimeError(err_preview), status_code=resp.status_code),
        )

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"{model_name} returned empty choices")
    message = choices[0].get("message") or {}
    raw_content = message.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        finish_reason = choices[0].get("finish_reason")
        raise ValueError(f"{model_name} returned empty content (finish_reason={finish_reason})")
    return raw_content


def generate_with_fallback(prompt):
    """Generates a validated English article payload strictly following MODEL_FALLBACK_CHAIN."""
    google_client = _resolve_google_client()
    nim_key = _resolve_nvidia_key()
    disabled_providers = set()
    missing_logged = set()

    if google_client is None:
        disabled_providers.add("google")
        missing_logged.add("google")
        print("ℹ️ Skipping Google Gemini/Gemma models (GEMINI_API_KEY / client not configured).")
    if not nim_key:
        disabled_providers.add("nvidia_nim")
        missing_logged.add("nvidia_nim")
        print("ℹ️ Skipping NVIDIA NIM models (NVIDIA_API_KEY not configured).")

    last_error = None
    total = len(MODEL_FALLBACK_CHAIN)

    for idx, entry in enumerate(MODEL_FALLBACK_CHAIN):
        provider = entry["provider"]
        model_name = entry["model"]
        wait_sec = entry.get("wait_sec", 2)

        if provider in disabled_providers:
            if provider not in missing_logged:
                print(f"⏭️ Skipping {model_name} (provider '{provider}' disabled after auth error).")
            continue

        try:
            print(f"Attempting to write English article using [{provider}] {model_name}...")
            if provider == "google":
                raw_text = _call_google_model(model_name, prompt, google_client)
            elif provider == "nvidia_nim":
                raw_text = _call_nvidia_nim_model(model_name, prompt, nim_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            article_data = validate_article_payload(parse_ai_json(raw_text))
            print(f"✅ Success with [{provider}] {model_name}!")
            return article_data

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"⚠️ [{provider}] {model_name} returned invalid/non-English JSON ({e}). Switching to backup model...")
        except Exception as e:
            last_error = e
            if _is_provider_auth_error(e):
                disabled_providers.add(provider)
                print(f"🔒 Provider '{provider}' credential error on {model_name} ({e}). Skipping remaining '{provider}' models.")
            else:
                print(f"⚠️ [{provider}] {model_name} busy/failed ({e}). Switching to backup model...")

        if idx < total - 1 and any(e["provider"] not in disabled_providers for e in MODEL_FALLBACK_CHAIN[idx + 1 :]):
            time.sleep(wait_sec)

    raise RuntimeError(f"All configured AI models failed. Last error: {last_error}")


def fetch_free_stock_image(query):
    """Fetches a royalty-free image URL & creator credit using Openverse."""
    try:
        url = "https://api.openverse.org/v1/images/"
        params = {"q": query, "page_size": 5, "aspect_ratio": "wide", "license_type": "commercial"}
        headers = {"User-Agent": "NewzireNewsBot/1.0"}
        res = requests.get(url, params=params, headers=headers, timeout=10)
        if res.status_code == 200:
            for img in res.json().get("results", []):
                img_url = img.get("url")
                creator = img.get("creator") or "Creative Commons"
                source = img.get("source") or "Openverse"
                if img_url and img_url.startswith("http"):
                    return img_url, f"Photo Credit: {creator} / {source.capitalize()} (Representational)"
    except Exception as e:
        print(f"Openverse stock fetch error: {e}")
    return None, None


def get_rss_image(entry):
    if "media_thumbnail" in entry and len(entry.media_thumbnail) > 0:
        return entry.media_thumbnail[0]["url"].replace("/240/", "/976/")
    if "media_content" in entry and len(entry.media_content) > 0:
        return entry.media_content[0]["url"]
    if "links" in entry:
        for link in entry.links:
            if link.get("type", "").startswith("image/"):
                return link.get("href")
    summary_html = entry.get("summary", "") or entry.get("description", "")
    if "<img" in summary_html:
        soup = BeautifulSoup(summary_html, "html.parser")
        img_tag = soup.find("img")
        if img_tag and img_tag.get("src", "").startswith("http"):
            return img_tag["src"]
    return None


# ==========================================================================
# SECURITY SANITIZATION HELPERS
# ==========================================================================

ALLOWED_ARTICLE_TAGS = {
    "p", "h2", "h3", "h4", "ul", "ol", "li", "strong", "b", "em", "i",
    "u", "blockquote", "br", "a", "code", "pre", "span", "hr",
}


def _sanitize_href_url(raw_url):
    cleaned = str(raw_url or "").strip()
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if lower.startswith(("javascript:", "data:", "vbscript:", "file:")):
        return ""
    if lower.startswith(("http://", "https://", "mailto:", "/", "#", "./", "../")):
        return cleaned
    if re.match(r"^[a-zA-Z0-9_\-./]+\.html(?:[#?][^\s\"'<>]*)?$", cleaned):
        return cleaned
    return ""


def _sanitize_article_html(raw_html):
    """Build-time allowlist HTML sanitizer blocking script/event/iframe XSS injection."""
    if not raw_html:
        return ""
    cleaned = re.sub(
        r"<\s*(script|style|iframe|object|embed|form|svg|math)\b[^>]*>.*?<\s*/\s*\1\s*>",
        "",
        str(raw_html),
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"<\s*(script|style|iframe|object|embed|form|input|button|textarea|meta|link|base)\b[^>]*/?>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    def _clean_tag(match):
        is_closing = bool(match.group(1))
        tag = match.group(2).lower()
        attrs_raw = match.group(3) or ""
        if tag not in ALLOWED_ARTICLE_TAGS:
            return ""
        if is_closing:
            return "" if tag in {"br", "hr"} else f"</{tag}>"
        if tag in {"br", "hr"}:
            return f"<{tag}>"
        safe_attrs = []
        if tag == "a":
            href_match = re.search(r"""href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", attrs_raw, re.IGNORECASE)
            if href_match:
                raw_href = href_match.group(1) or href_match.group(2) or href_match.group(3) or ""
                safe_href = _sanitize_href_url(html.unescape(raw_href))
                if safe_href:
                    safe_attrs.append(f'href="{html.escape(safe_href, quote=True)}"')
                    if safe_href.startswith(("http://", "https://")):
                        safe_attrs.append('target="_blank"')
                        safe_attrs.append('rel="noopener noreferrer"')
        return f"<{tag} {' '.join(safe_attrs)}>" if safe_attrs else f"<{tag}>"

    return re.sub(r"<(/?)([A-Za-z0-9]+)([^>]*)>", _clean_tag, cleaned).strip()


# ==========================================================================
# EDITORIAL TEXT & ARTICLE SELECTION HELPERS
# ==========================================================================

def _extract_article_excerpt(article, max_chars=210):
    raw_html = str(article.get("content_html", "") or "")
    plain = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) <= max_chars:
        return plain
    trimmed = plain[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{trimmed}..."


def _estimate_read_minutes(article):
    raw_html = str(article.get("content_html", "") or "")
    plain = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
    words = len(re.findall(r"\S+", plain))
    return max(2, min(12, round(words / 180)))


def _article_matches_category(article, cat_key, config):
    art_cat = str(article.get("category", "") or "").strip().lower()
    target = str(cat_key or "").strip().lower()
    if not target or target == "all":
        return True
    if art_cat == target:
        return True
    for cat in config.get("categories", []):
        cat_id = str(cat.get("id", "")).strip().lower()
        cat_name = str(cat.get("name", "")).strip().lower()
        aliases = {str(a).strip().lower() for a in cat.get("aliases", []) if str(a).strip()} | {cat_id, cat_name}
        if target in aliases:
            return art_cat in aliases
    return False


def _filter_by_categories(articles, cat_keys, config, limit=4, exclude_slugs=None):
    exclude = exclude_slugs if exclude_slugs is not None else set()
    matched = []
    for art in articles:
        slug = clean_slug(art.get("slug", ""))
        if slug in exclude:
            continue
        if any(_article_matches_category(art, k, config) for k in cat_keys):
            matched.append(art)
            if len(matched) >= limit:
                break
    return matched


def _category_href(cat_name_or_id, config, prefix=""):
    target = str(cat_name_or_id or "").strip().lower()
    for cat in config.get("categories", []):
        aliases = {str(a).strip().lower() for a in cat.get("aliases", [])} | {
            str(cat.get("id", "")).strip().lower(),
            str(cat.get("name", "")).strip().lower(),
        }
        if target in aliases:
            return f"{prefix}{os.path.basename(cat['file'])}"
    return f"{prefix}india.html"


# ==========================================================================
# SHARED CHROME: <HEAD>, TOP STRIP, STICKY TWO-TIER NAVBAR, SUBNAV, FOOTER
# ==========================================================================

def get_head_links(css_path="style.css", preload_img=None, config=None):
    """Common <head> tags: Anti-FOUC theme init, Google Fonts (Inter + Newsreader), optional LCP preload, CSS."""
    escaped_preload = html.escape(preload_img, quote=True) if preload_img else ""
    preload_tag = (
        f'    <link rel="preload" as="image" href="{escaped_preload}" fetchpriority="high">\n'
        if escaped_preload
        else ""
    )
    header_cfg = config.get("header") if isinstance(config, dict) and isinstance(config.get("header"), dict) else {}
    default_mode = str(header_cfg.get("default_color_mode", "dark") or "dark").strip().lower()
    if default_mode not in VALID_COLOR_MODES:
        default_mode = "dark"
    toggle_enabled = bool(header_cfg.get("theme_toggle_enabled", True))
    toggle_js = "true" if toggle_enabled else "false"
    theme_init_script = (
        f"    <script>(function(){{var d='{default_mode}';var e={toggle_js};var s=null;"
        "if(e){try{s=localStorage.getItem('newzire-theme');}catch(err){}}"
        "var m=(s==='light'||s==='dark')?s:(d==='system'?((window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light'):d);"
        "document.documentElement.setAttribute('data-theme',m==='dark'?'dark':'light');"
        "document.documentElement.setAttribute('data-default-theme',d);"
        f"document.documentElement.setAttribute('data-theme-toggle-enabled','{toggle_js}');"
        "})();</script>"
    )
    return f"""    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 rx=%2218%22 fill=%22%230e0e0e%22/><rect x=%2216%22 y=%2222%22 width=%2210%22 height=%2256%22 rx=%224%22 fill=%22%23ff542e%22/><text x=%2258%%22 y=%2254%%22 dominant-baseline=%22central%22 text-anchor=%22middle%22 fill=%22%23f7f5f2%22 font-family=%22Inter,sans-serif%22 font-weight=%22900%22 font-size=%2254%22>N</text></svg>">
{theme_init_script}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="preconnect" href="https://wsrv.nl" crossorigin>
{preload_tag}    <link rel="stylesheet" href="{css_path}">
    <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,wght@0,600;1,500&display=swap" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,wght@0,600;1,500&display=swap" rel="stylesheet"></noscript>"""


def get_top_bar_html(config, today_str, prefix=""):
    header_cfg = config.get("header", {})
    if not header_cfg.get("top_bar_enabled", True):
        return ""
    edition = html.escape(str(header_cfg.get("edition_tag", "DIGITAL EDITION • INDIA & WORLD")), quote=True)
    desk_note = html.escape(str(header_cfg.get("top_bar_text", "New Delhi • Global Desk")), quote=True)
    date_part = (
        f'<span class="utility-date">{html.escape(today_str)}</span>'
        if header_cfg.get("top_bar_show_date", True)
        else ""
    )
    sep_part = '<span class="utility-sep">•</span>' if date_part and desk_note else ""
    return f"""  <div class="top-utility-bar">
    <div class="wide-wrap top-utility-inner">
      <div class="utility-left">
        <svg width="14" height="14" style="color:var(--accent);flex-shrink:0;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
          <line x1="16" y1="2" x2="16" y2="6"></line>
          <line x1="8" y1="2" x2="8" y2="6"></line>
          <line x1="3" y1="10" x2="21" y2="10"></line>
        </svg>
        {date_part}
        {sep_part}
        <span class="utility-location">{desk_note}</span>
      </div>
      <div class="utility-right">
        <span class="edition-badge-text">{edition}</span>
        <div class="utility-socials">
          <a href="https://facebook.com/NewzireNews" target="_blank" rel="noopener noreferrer" aria-label="Facebook" class="utility-social-link" title="Facebook">
            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M14 13.5h2.5l1-4H14v-2c0-1.03 0-2 2-2h1.5V2.14c-.326-.043-1.557-.14-2.857-.14C11.928 2 10 3.657 10 6.7v2.8H7v4h3V22h4v-8.5z"/></svg>
          </a>
          <a href="https://x.com/newzire_in" target="_blank" rel="noopener noreferrer" aria-label="X" class="utility-social-link" title="X (Twitter)">
            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
          </a>
          <a href="https://instagram.com/newzire.news" target="_blank" rel="noopener noreferrer" aria-label="Instagram" class="utility-social-link" title="Instagram">
            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
          </a>
        </div>
        <span class="utility-sep">|</span>
        <button type="button" onclick="openContactModal()" class="utility-contact-btn">Contact Us</button>
      </div>
    </div>
  </div>"""


def get_header_html(config, prefix="", active_file=""):
    header_cfg = config.get("header", {})
    if not header_cfg.get("enabled", True):
        return ""

    home_href = f"{prefix}index.html"
    p = prefix

    def _active_cls(files):
        return " is-active" if active_file in files else ""

    def _active_bar(files):
        return '<span class="nav-active-bar"></span>' if active_file in files else ""

    chevron_svg = (
        '<svg class="nav-chevron" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.4" aria-hidden="true">'
        '<path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/></svg>'
    )

    toggle_html = ""
    if header_cfg.get("theme_toggle_enabled", True):
        toggle_html = """        <button type="button" id="theme-toggle-btn" onclick="toggleTheme()" aria-label="Toggle Light or Dark Mode" class="theme-toggle-btn">
          <span id="theme-icon" class="theme-icon-sun" aria-hidden="true">☀</span>
          <span id="theme-label">Light</span>
        </button>"""

    # Build Quick Desk Sub-Header dynamically from config categories
    subnav_parts = []
    cats = config.get("categories", [])
    for idx, cat in enumerate(cats):
        cat_file = os.path.basename(cat.get("file", ""))
        cat_name = html.escape(cat.get("name", ""), quote=True)
        is_act = " is-active" if active_file == cat_file else ""
        if idx > 0:
            subnav_parts.append('<span class="subnav-dot" aria-hidden="true">•</span>')
        subnav_parts.append(f'<a href="{p}{cat_file}" class="subnav-link{is_act}">{cat_name}</a>')
    subnav_parts.append('<span class="subnav-dot" aria-hidden="true">•</span>')
    subnav_parts.append(
        f'<a href="{p}index.html#latest-trending-box" class="subnav-link">'
        '<span class="dot-accent"></span><span>Trending</span></a>'
    )
    subnav_links_html = "\n        ".join(subnav_parts)

    return f"""  <header class="site-header">
    <div class="wide-wrap header-inner">
      <a href="{home_href}" class="brand-wordmark" aria-label="Newzire Home">
        <span class="brand-accent-bar" aria-hidden="true"></span>
        <span class="brand-main">Newz</span><span class="brand-highlight">ire</span><span class="brand-dot" aria-hidden="true"></span>
      </a>

      <nav aria-label="Main Navigation" class="desktop-nav">
        <a href="{home_href}" class="nav-top-link{_active_cls({'index.html'})}">
          <span>Home</span>
          {_active_bar({'index.html'})}
        </a>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false"
                  class="nav-top-link{_active_cls({'india.html', 'world.html', 'politics.html', 'sports.html'})}">
            <span>News</span>
            {chevron_svg}
            {_active_bar({'india.html', 'world.html', 'politics.html', 'sports.html'})}
          </button>
          <div class="nav-dropdown-panel w-2col">
            <div class="dropdown-grid-2">
              <div>
                <p class="dropdown-heading">Coverage</p>
                <div class="dropdown-list">
                  <a href="{p}india.html" class="dropdown-link">
                    <span>India <span class="dropdown-link-sub">(National)</span></span>
                    <span class="dropdown-arrow">→</span>
                  </a>
                  <a href="{p}world.html" class="dropdown-link">
                    <span>World <span class="dropdown-link-sub">(Global)</span></span>
                    <span class="dropdown-arrow">→</span>
                  </a>
                </div>
              </div>
              <div class="dropdown-col-divider">
                <p class="dropdown-heading">Beats</p>
                <div class="dropdown-list">
                  <a href="{p}politics.html" class="dropdown-link">
                    <span>Politics &amp; Policy</span>
                    <span class="dropdown-arrow">→</span>
                  </a>
                  <a href="{p}sports.html" class="dropdown-link">
                    <span>Sports</span>
                    <span class="dropdown-arrow">→</span>
                  </a>
                  <a href="{p}index.html#latest-trending-box" class="dropdown-link">
                    <span style="display:inline-flex;align-items:center;gap:6px;"><span class="dot-accent"></span>Trending</span>
                    <span class="dropdown-arrow">→</span>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false"
                  class="nav-top-link{_active_cls({'technology.html', 'economy.html'})}">
            <span>Tech &amp; Business</span>
            <span class="nav-dot-amber" aria-hidden="true"></span>
            {chevron_svg}
            {_active_bar({'technology.html', 'economy.html'})}
          </button>
          <div class="nav-dropdown-panel w-1col">
            <p class="dropdown-heading">Innovation &amp; Markets</p>
            <div class="dropdown-list">
              <a href="{p}technology.html" class="dropdown-link">
                <span>Technology &amp; AI</span>
                <span class="dropdown-pill accent">AI</span>
              </a>
              <a href="{p}economy.html" class="dropdown-link">
                <span>Economy &amp; Markets</span>
                <span class="dropdown-arrow">→</span>
              </a>
              <a href="{p}index.html#markets-desk-box" class="dropdown-link">
                <span>Markets &amp; Explainers</span>
                <span class="dropdown-pill neutral">Deep Dive</span>
              </a>
            </div>
          </div>
        </div>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false"
                  class="nav-top-link{_active_cls({'opinion.html', 'editorial-policy.html'})}">
            <span>Views &amp; Specials</span>
            {chevron_svg}
            {_active_bar({'opinion.html'})}
          </button>
          <div class="nav-dropdown-panel w-1col">
            <p class="dropdown-heading">Editorial &amp; Specials</p>
            <div class="dropdown-list">
              <a href="{p}opinion.html" class="dropdown-link"><span>Opinion &amp; Editorials</span><span class="dropdown-arrow">→</span></a>
              <a href="{p}index.html#views-column" class="dropdown-link"><span>Strategic Analysis</span><span class="dropdown-arrow">→</span></a>
              <a href="{p}about.html" class="dropdown-link"><span>About Newzire</span><span class="dropdown-arrow">→</span></a>
              <div class="dropdown-divider"></div>
              <a href="{p}editorial-policy.html" class="dropdown-link">
                <span>Fact-Lock Standards</span>
                <span class="dropdown-pill accent">Policy</span>
              </a>
            </div>
          </div>
        </div>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false"
                  class="nav-top-link{_active_cls({'culture.html'})}">
            <span>Culture &amp; Life</span>
            {chevron_svg}
            {_active_bar({'culture.html'})}
          </button>
          <div class="nav-dropdown-panel w-2col align-right">
            <div class="dropdown-grid-2">
              <div>
                <p class="dropdown-heading">Arts</p>
                <div class="dropdown-list">
                  <a href="{p}culture.html" class="dropdown-link"><span>Culture Desk</span><span class="dropdown-arrow">→</span></a>
                  <a href="{p}index.html#culture-column" class="dropdown-link"><span>Film &amp; Cinema</span><span class="dropdown-arrow">→</span></a>
                  <a href="{p}index.html#culture-column" class="dropdown-link"><span>Books &amp; Heritage</span><span class="dropdown-arrow">→</span></a>
                </div>
              </div>
              <div class="dropdown-col-divider">
                <p class="dropdown-heading">Living</p>
                <div class="dropdown-list">
                  <a href="{p}index.html#culture-column" class="dropdown-link"><span>Lifestyle &amp; Food</span><span class="dropdown-arrow">→</span></a>
                  <a href="{p}index.html#culture-column" class="dropdown-link"><span>Travel &amp; Design</span><span class="dropdown-arrow">→</span></a>
                  <button type="button" onclick="openServiceModal('health-wellness')" class="dropdown-link"><span>Wellness Guide</span><span class="dropdown-arrow">→</span></button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false" class="nav-top-link">
            <span>Services</span>
            {chevron_svg}
          </button>
          <div class="nav-dropdown-panel w-2col align-right">
            <div class="dropdown-grid-2">
              <div>
                <p class="dropdown-heading">Advice Desks</p>
                <div class="dropdown-list">
                  <button type="button" onclick="openServiceModal('health-wellness')" class="dropdown-link"><span>Health &amp; Wellness</span></button>
                  <button type="button" onclick="openServiceModal('legal-consumer')" class="dropdown-link"><span>Legal &amp; Consumer</span></button>
                  <button type="button" onclick="openServiceModal('fact-check')" class="dropdown-link"><span>Fact-Check Desk</span></button>
                </div>
              </div>
              <div class="dropdown-col-divider">
                <p class="dropdown-heading">Reader Utility</p>
                <div class="dropdown-list">
                  <button type="button" onclick="openServiceModal('career-exams')" class="dropdown-link"><span>Career &amp; Exams</span></button>
                  <button type="button" onclick="openServiceModal('jobs-desk')" class="dropdown-link"><span>Jobs Desk</span></button>
                  <button type="button" onclick="openServiceModal('events-calendar')" class="dropdown-link"><span>Events Calendar</span></button>
                  <button type="button" onclick="openServiceModal('civic-directory')" class="dropdown-link"><span>Civic Directory</span></button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="nav-item-group" data-dropdown>
          <button type="button" onclick="toggleDropdown(this)" aria-expanded="false"
                  class="nav-top-link{_active_cls({'contact.html'})}">
            <span>Media</span>
            {chevron_svg}
            {_active_bar({'contact.html'})}
          </button>
          <div class="nav-dropdown-panel w-1col align-right">
            <p class="dropdown-heading">Multimedia &amp; More</p>
            <div class="dropdown-list">
              <a href="{p}index.html#visual-markets-section" class="dropdown-link"><span>Videos &amp; Briefings</span></a>
              <a href="{p}index.html#visual-markets-section" class="dropdown-link"><span>Photo Essays</span></a>
              <button type="button" onclick="openServiceModal('archives')" class="dropdown-link"><span>Archives</span></button>
              <div class="dropdown-divider"></div>
              <a href="{p}contact.html" class="dropdown-link" style="color:var(--accent);font-weight:600;"><span>Contact Us</span></a>
            </div>
          </div>
        </div>
      </nav>

      <div class="header-controls">
{toggle_html}
        <button type="button" id="mobile-menu-open-btn" onclick="openMobileDrawer()" aria-label="Open menu" class="mobile-menu-btn">
          <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
          </svg>
        </button>
      </div>
    </div>
  </header>

  <nav aria-label="Quick Desk Categories" class="subnav-bar">
    <div class="wide-wrap subnav-inner no-scrollbar">
      <span class="subnav-label">Quick Desk:</span>
      <div class="subnav-links">
        {subnav_links_html}
      </div>
    </div>
  </nav>"""


def get_footer_html(config, prefix="", active_file=""):
    footer_cfg = config.get("footer", {})
    if not footer_cfg.get("enabled", True):
        return ""
    p = prefix
    year = datetime.now().year
    brand_tagline = html.escape(str(footer_cfg.get("brand_tagline", "")), quote=True)
    copyright_raw = str(footer_cfg.get("copyright_text", "© {year} Newzire Digital Media. All Rights Reserved."))
    copyright_text = html.escape(copyright_raw.replace("{year}", str(year)), quote=True)

    social_links_html = []
    for soc in footer_cfg.get("social_links", []):
        lbl = html.escape(str(soc.get("label", "")), quote=True)
        href = _sanitize_href_url(soc.get("href", "")) or "#"
        target = ' target="_blank" rel="noopener noreferrer"' if soc.get("new_tab", True) and href.startswith("http") else ""
        social_links_html.append(f'<a href="{html.escape(href, quote=True)}"{target} class="footer-social-pill">{lbl}</a>')

    cols_html = []
    for col in footer_cfg.get("columns", [])[:3]:
        col_title = html.escape(str(col.get("title", "")), quote=True)
        links_li = []
        for lnk in col.get("links", []):
            lbl = html.escape(str(lnk.get("label", "")), quote=True)
            raw_href = str(lnk.get("href", "")).strip()
            if raw_href.startswith(("http://", "https://", "mailto:", "#")):
                href = _sanitize_href_url(raw_href) or "#"
            else:
                href = f"{p}{raw_href.lstrip('/')}"
            links_li.append(f'<li><a href="{html.escape(href, quote=True)}">{lbl}</a></li>')
        cols_html.append(
            f"""        <div>
          <h4 class="footer-col-title">{col_title}</h4>
          <ul class="footer-links">
            {"".join(links_li)}
          </ul>
        </div>"""
        )

    return f"""  <footer class="site-footer">
    <div class="site-wrap footer-inner">
      <div class="footer-grid">
        <div>
          <a href="{p}index.html" class="brand-wordmark" style="font-size:24px;" aria-label="Newzire">
            <span class="brand-accent-bar" style="height:22px;width:4px;margin-right:7px;"></span>
            <span class="brand-main" style="font-size:26px;">Newz</span><span class="brand-highlight" style="font-size:26px;">ire</span><span class="brand-dot" style="width:5px;height:5px;"></span>
          </a>
          <p class="footer-brand-desc">{brand_tagline}</p>
          <div class="footer-socials">
            {"".join(social_links_html)}
          </div>
        </div>
{chr(10).join(cols_html)}
      </div>

      <div class="footer-bottom">
        <p>{copyright_text}</p>
        <div class="footer-bottom-links">
          <a href="{p}editorial-policy.html">Editorial Policy</a>
          <a href="{p}privacy-policy.html">Privacy</a>
          <button type="button" onclick="openContactModal()">Advertise / Partner With Us</button>
        </div>
      </div>
    </div>
  </footer>"""


def get_mobile_drawer_and_modals_html(config, prefix=""):
    p = prefix
    promo = config.get("promo_banner", DEFAULT_SITE_CONFIG["promo_banner"])
    services_json = json.dumps(READER_SERVICE_DESKS, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    promo_json = json.dumps(promo, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")

    return f"""  <div id="mobile-drawer-overlay" onclick="closeMobileDrawer()" class="modal-backdrop hidden"></div>
  <aside id="mobile-drawer" aria-label="Mobile Navigation" class="mobile-drawer">
    <div class="mobile-drawer-head">
      <a href="{p}index.html" class="brand-wordmark" aria-label="Newzire">
        <span class="brand-accent-bar" style="height:20px;width:4px;margin-right:7px;"></span>
        <span class="brand-main" style="font-size:22px;">Newz</span><span class="brand-highlight" style="font-size:22px;">ire</span>
      </a>
      <button type="button" onclick="closeMobileDrawer()" aria-label="Close Menu" class="modal-close-btn">✕</button>
    </div>

    <div class="mobile-drawer-body">
      <a href="{p}index.html" class="mobile-home-link">Home — Frontpage</a>

      <div class="mobile-acc">
        <button type="button" onclick="toggleMobileAccordion(this)" class="mobile-acc-btn">
          <span>News &amp; Beats</span><span class="acc-icon">▾</span>
        </button>
        <div class="mobile-acc-panel hidden">
          <a href="{p}india.html">India (National)</a>
          <a href="{p}world.html">World (Global)</a>
          <a href="{p}politics.html">Politics &amp; Policy</a>
          <a href="{p}sports.html">Sports</a>
          <a href="{p}index.html#latest-trending-box" style="color:var(--accent);font-weight:600;">Trending</a>
        </div>
      </div>

      <div class="mobile-acc">
        <button type="button" onclick="toggleMobileAccordion(this)" class="mobile-acc-btn">
          <span>Tech &amp; Business</span><span class="acc-icon">▾</span>
        </button>
        <div class="mobile-acc-panel hidden">
          <a href="{p}technology.html">Technology &amp; AI</a>
          <a href="{p}economy.html">Economy &amp; Markets</a>
          <a href="{p}index.html#markets-desk-box">Explainers</a>
        </div>
      </div>

      <div class="mobile-acc">
        <button type="button" onclick="toggleMobileAccordion(this)" class="mobile-acc-btn">
          <span>Views &amp; Culture</span><span class="acc-icon">▾</span>
        </button>
        <div class="mobile-acc-panel hidden">
          <a href="{p}opinion.html">Opinion &amp; Analysis</a>
          <a href="{p}culture.html">Culture &amp; Life</a>
          <a href="{p}about.html">About Newzire</a>
          <a href="{p}editorial-policy.html">Editorial Policy</a>
        </div>
      </div>

      <div class="mobile-acc">
        <button type="button" onclick="toggleMobileAccordion(this)" class="mobile-acc-btn">
          <span>Reader Services</span><span class="acc-icon">▾</span>
        </button>
        <div class="mobile-acc-panel hidden">
          <button type="button" onclick="closeMobileDrawer();openServiceModal('health-wellness')">Health &amp; Wellness</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('legal-consumer')">Legal &amp; Consumer</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('career-exams')">Career &amp; Exams</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('jobs-desk')">Jobs Desk</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('events-calendar')">Events Calendar</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('civic-directory')">Civic Directory</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('fact-check')">Fact-Check Desk</button>
          <button type="button" onclick="closeMobileDrawer();openServiceModal('archives')">Archives</button>
        </div>
      </div>

      <div class="mobile-acc">
        <button type="button" onclick="toggleMobileAccordion(this)" class="mobile-acc-btn">
          <span>Media &amp; Contact</span><span class="acc-icon">▾</span>
        </button>
        <div class="mobile-acc-panel hidden">
          <a href="{p}index.html#visual-markets-section">Videos &amp; Photos</a>
          <a href="{p}contact.html" style="color:var(--accent);font-weight:600;">Contact Us</a>
          <a href="{p}privacy-policy.html">Privacy Policy</a>
        </div>
      </div>
    </div>
  </aside>

  <div id="service-modal" class="modal-shell hidden" onclick="if(event.target===this) closeModal('service-modal')">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="service-modal-title">
      <div class="modal-header">
        <div>
          <span id="service-modal-badge" class="section-kicker" style="font-size:10px;">Reader Desk</span>
          <h3 id="service-modal-title" class="panel-title" style="font-size:17px;">Service Details</h3>
        </div>
        <button type="button" onclick="closeModal('service-modal')" aria-label="Close modal" class="modal-close-btn">✕</button>
      </div>
      <div id="service-modal-content" class="modal-body"></div>
    </div>
  </div>

  <div id="contact-modal" class="modal-shell hidden" onclick="if(event.target===this) closeModal('contact-modal')">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="contact-modal-title">
      <div class="modal-header">
        <div>
          <span class="section-kicker" style="font-size:10px;">Editorial &amp; Partnerships</span>
          <h3 id="contact-modal-title" class="panel-title" style="font-size:17px;">Contact Newzire Newsroom</h3>
        </div>
        <button type="button" onclick="closeModal('contact-modal')" aria-label="Close contact dialog" class="modal-close-btn">✕</button>
      </div>
      <div class="modal-body">
        <p>Reach our editorial desk for verified news tips, factual corrections, press releases, or partnership opportunities.</p>
        <form onsubmit="handleContactSubmit(event)" class="tip-form">
          <input type="text" required placeholder="Your Name *" class="form-input">
          <input type="email" required placeholder="Email Address *" class="form-input">
          <textarea rows="3" required placeholder="Share your message, tip, or inquiry..." class="form-input"></textarea>
          <div class="modal-actions">
            <a href="{p}contact.html" class="btn-secondary">Full Contact Page</a>
            <button type="submit" class="btn-accent">Send Message →</button>
          </div>
        </form>
      </div>
    </div>
  </div>

  <div id="toast-box" class="toast-box" role="status" aria-live="polite">
    <span class="dot-accent"></span>
    <span id="toast-text">Notification</span>
  </div>

  <script>
  (function () {{
    var SERVICE_DESKS = {services_json};
    var PROMO_BANNER = {promo_json};
    var toastTimer = null;

    function syncThemeUI(theme) {{
      var icon = document.getElementById('theme-icon');
      var label = document.getElementById('theme-label');
      if (!icon || !label) return;
      if (theme === 'light') {{
        icon.textContent = '☾';
        icon.className = 'theme-icon-moon';
        label.textContent = 'Dark';
      }} else {{
        icon.textContent = '☀';
        icon.className = 'theme-icon-sun';
        label.textContent = 'Light';
      }}
    }}

    window.applyTheme = function (theme) {{
      var resolved = theme === 'light' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', resolved);
      syncThemeUI(resolved);
      try {{ localStorage.setItem('newzire-theme', resolved); }} catch (e) {{}}
    }};

    window.toggleTheme = function () {{
      var current = document.documentElement.getAttribute('data-theme') || 'dark';
      window.applyTheme(current === 'dark' ? 'light' : 'dark');
    }};

    window.closeAllDropdowns = function () {{
      document.querySelectorAll('[data-dropdown]').forEach(function (g) {{
        g.classList.remove('is-open');
        var b = g.querySelector('button[aria-expanded]');
        if (b) b.setAttribute('aria-expanded', 'false');
      }});
    }};

    window.toggleDropdown = function (btn) {{
      var group = btn.closest('[data-dropdown]');
      if (!group) return;
      var wasOpen = group.classList.contains('is-open');
      window.closeAllDropdowns();
      if (!wasOpen) {{
        group.classList.add('is-open');
        btn.setAttribute('aria-expanded', 'true');
      }}
    }};

    document.addEventListener('click', function (e) {{
      if (!e.target.closest('[data-dropdown]')) window.closeAllDropdowns();
    }});

    document.addEventListener('keydown', function (e) {{
      if (e.key === 'Escape') {{
        window.closeAllDropdowns();
        window.closeModal('service-modal');
        window.closeModal('contact-modal');
        window.closeMobileDrawer();
      }}
    }});

    window.openMobileDrawer = function () {{
      var ov = document.getElementById('mobile-drawer-overlay');
      var dr = document.getElementById('mobile-drawer');
      if (!ov || !dr) return;
      ov.classList.remove('hidden');
      requestAnimationFrame(function () {{ dr.classList.add('is-open'); }});
      document.body.style.overflow = 'hidden';
    }};

    window.closeMobileDrawer = function () {{
      var ov = document.getElementById('mobile-drawer-overlay');
      var dr = document.getElementById('mobile-drawer');
      if (!ov || !dr) return;
      dr.classList.remove('is-open');
      setTimeout(function () {{ ov.classList.add('hidden'); }}, 260);
      document.body.style.overflow = '';
    }};

    window.toggleMobileAccordion = function (btn) {{
      var panel = btn.nextElementSibling;
      var icon = btn.querySelector('.acc-icon');
      if (!panel) return;
      var willOpen = panel.classList.contains('hidden');
      panel.classList.toggle('hidden', !willOpen);
      if (icon) icon.style.transform = willOpen ? 'rotate(180deg)' : 'rotate(0deg)';
    }};

    window.openModalElement = function (id) {{
      var m = document.getElementById(id);
      if (!m) return;
      m.classList.remove('hidden');
      document.body.style.overflow = 'hidden';
    }};

    window.closeModal = function (id) {{
      var m = document.getElementById(id);
      if (!m) return;
      m.classList.add('hidden');
      document.body.style.overflow = '';
    }};

    window.showToast = function (msg) {{
      var box = document.getElementById('toast-box');
      var txt = document.getElementById('toast-text');
      if (!box || !txt) return;
      txt.textContent = msg;
      box.classList.add('is-visible');
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(function () {{ box.classList.remove('is-visible'); }}, 3200);
    }};

    window.copyTextToClipboard = function (text, successMsg) {{
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(function () {{
          window.showToast(successMsg || 'Copied to clipboard!');
        }}).catch(function () {{ fallbackCopy(text, successMsg); }});
        return;
      }}
      fallbackCopy(text, successMsg);
    }};

    function fallbackCopy(text, successMsg) {{
      try {{
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        window.showToast(successMsg || 'Copied to clipboard!');
      }} catch (err) {{
        window.showToast('Unable to copy automatically.');
      }}
    }}

    function escHtml(val) {{
      return String(val == null ? '' : val)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    window.openServiceModal = function (key) {{
      window.closeAllDropdowns();
      var desk = SERVICE_DESKS[key];
      if (!desk) return;
      document.getElementById('service-modal-badge').textContent = desk.badge || '';
      document.getElementById('service-modal-title').textContent = desk.title || '';
      var items = (desk.highlights || []).map(function (h) {{
        return '<div class="modal-highlight-item"><span class="dot-accent" style="margin-top:6px;"></span><span>' + escHtml(h) + '</span></div>';
      }}).join('');
      var bodyEl = document.getElementById('service-modal-content');
      bodyEl.innerHTML =
        '<p style="color:var(--text-primary);line-height:1.65;">' + escHtml(desk.summary || '') + '</p>' +
        '<div style="display:flex;flex-direction:column;gap:8px;">' + items + '</div>' +
        '<div class="modal-actions">' +
          '<button type="button" data-modal-close class="btn-secondary">Close</button>' +
          '<button type="button" data-modal-cta class="btn-accent">' + escHtml(desk.cta || 'Continue') + ' →</button>' +
        '</div>';
      var closeBtn = bodyEl.querySelector('[data-modal-close]');
      var ctaBtn = bodyEl.querySelector('[data-modal-cta]');
      if (closeBtn) closeBtn.addEventListener('click', function () {{ window.closeModal('service-modal'); }});
      if (ctaBtn) ctaBtn.addEventListener('click', function () {{
        window.closeModal('service-modal');
        window.showToast((desk.title || 'Reader Desk') + ' — Desk request noted');
      }});
      window.openModalElement('service-modal');
    }};

    window.openDealModal = function () {{
      window.closeAllDropdowns();
      var code = String(PROMO_BANNER.promo_code || 'NEWZIRE30');
      document.getElementById('service-modal-badge').textContent = PROMO_BANNER.badge || 'SPONSORED • DEAL OF THE DAY';
      document.getElementById('service-modal-title').textContent = PROMO_BANNER.modal_title || PROMO_BANNER.headline || '';
      var bodyEl = document.getElementById('service-modal-content');
      bodyEl.innerHTML =
        '<p style="color:var(--text-primary);line-height:1.65;">' + escHtml(PROMO_BANNER.modal_summary || '') + '</p>' +
        '<div class="modal-promo-box">' +
          '<div><p style="font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);">Promo Code</p>' +
          '<p style="font-size:18px;font-weight:800;letter-spacing:0.12em;color:var(--accent);">' + escHtml(code) + '</p></div>' +
          '<button type="button" data-copy-promo class="btn-accent">Copy Code</button>' +
        '</div>' +
        '<p style="font-size:12px;color:var(--text-muted);">' + escHtml(PROMO_BANNER.footnote || '') + '</p>';
      var copyPromoBtn = bodyEl.querySelector('[data-copy-promo]');
      if (copyPromoBtn) copyPromoBtn.addEventListener('click', function () {{
        window.copyTextToClipboard(code, 'Promo code ' + code + ' copied!');
      }});
      window.openModalElement('service-modal');
    }};

    window.openContactModal = function () {{
      window.closeAllDropdowns();
      window.openModalElement('contact-modal');
    }};

    window.handleContactSubmit = function (evt) {{
      evt.preventDefault();
      evt.target.reset();
      window.closeModal('contact-modal');
      window.showToast('Message received by the Newzire newsroom. Thank you!');
    }};

    window.handleTipSubmit = function (evt) {{
      evt.preventDefault();
      var nameEl = document.getElementById('tip-name');
      var topicEl = document.getElementById('tip-topic');
      var successBox = document.getElementById('tip-success-state');
      var sender = nameEl ? nameEl.value.trim() : '';
      var topic = topicEl ? topicEl.value : 'News Tip';
      evt.target.reset();
      if (successBox) {{
        successBox.classList.remove('hidden');
        setTimeout(function () {{ successBox.classList.add('hidden'); }}, 6000);
      }}
      window.showToast('Thank you' + (sender ? ', ' + sender : '') + '! Your ' + topic + ' has been logged for our editors.');
    }};

    window.switchTabGroup = function (groupName, activeKey) {{
      document.querySelectorAll('[data-tab-btn="' + groupName + '"]').forEach(function (btn) {{
        btn.classList.toggle('is-active', btn.getAttribute('data-tab-key') === activeKey);
      }});
      document.querySelectorAll('[data-tab-panel="' + groupName + '"]').forEach(function (panel) {{
        panel.classList.toggle('hidden', panel.getAttribute('data-tab-key') !== activeKey);
      }});
    }};

    var tickerPaused = false;
    var tickerShift = 0;

    window.toggleTickerPause = function () {{
      var track = document.getElementById('ticker-track');
      var icon = document.getElementById('ticker-pause-icon');
      if (!track || !icon) return;
      tickerPaused = !tickerPaused;
      track.classList.toggle('is-paused', tickerPaused);
      if (tickerPaused) {{
        icon.innerHTML = '<path d="M8 5v14l11-7z"/>';
        window.showToast('Breaking news ticker paused');
      }} else {{
        icon.innerHTML = '<path d="M6 5h4v14H6zm8 0h4v14h-4z"/>';
        window.showToast('Breaking news ticker resumed');
      }}
    }};

    window.nextTickerHeadline = function () {{
      var track = document.getElementById('ticker-track');
      if (!track) return;
      var items = Array.prototype.slice.call(track.querySelectorAll('.ticker-item'));
      var half = Math.floor(items.length / 2);
      if (half < 2) return;
      tickerShift = (tickerShift + 1) % half;
      var originals = items.slice(0, half);
      var rotated = originals.slice(tickerShift).concat(originals.slice(0, tickerShift));
      var htmlStr = rotated.concat(rotated).map(function (el) {{ return el.outerHTML; }}).join('');
      track.innerHTML = htmlStr;
      track.style.animation = 'none';
      void track.offsetWidth;
      track.style.animation = '';
    }};

    window.copyCurrentPageLink = function () {{
      var btn = document.getElementById('copy-link-btn');
      window.copyTextToClipboard(window.location.href, 'Story link copied to clipboard!');
      if (btn) {{
        btn.textContent = 'Copied ✓';
        setTimeout(function () {{ btn.textContent = 'Copy Link'; }}, 2000);
      }}
    }};

    document.addEventListener('DOMContentLoaded', function () {{
      var cur = document.documentElement.getAttribute('data-theme') || 'dark';
      syncThemeUI(cur);
    }});
  }})();
  </script>"""


# ==========================================================================
# NEWZIRE 5-BAND HOMEPAGE BUILDER (index.html)
# ==========================================================================

def build_ticker_html(articles):
    ticker_items = articles[:6]
    if not ticker_items:
        return ""
    links = []
    for art in ticker_items:
        slug = clean_slug(art.get("slug", "story"))
        cat = html.escape(str(art.get("category", "News")), quote=True)
        title = html.escape(str(art.get("title", "")), quote=True)
        links.append(
            f'<a href="news/{slug}.html" class="ticker-item">'
            f'<span class="ticker-tag">{cat}</span>'
            f'<span>{title}</span>'
            f'<span class="ticker-sep" aria-hidden="true">•</span>'
            f'</a>'
        )
    doubled = "\n          ".join(links + links)
    return f"""  <div class="ticker-bar">
    <div class="site-wrap ticker-inner">
      <div class="ticker-badge">
        <span class="pulse-dot"></span>
        <span class="ticker-badge-text">Live Updates</span>
      </div>
      <div class="ticker-wrapper">
        <div id="ticker-track" class="ticker-track">
          {doubled}
        </div>
      </div>
      <div class="ticker-controls">
        <button type="button" id="ticker-pause-btn" onclick="toggleTickerPause()" title="Pause / Play Ticker" aria-label="Pause or play breaking news" class="ticker-ctrl-btn">
          <svg id="ticker-pause-icon" width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>
        </button>
        <button type="button" onclick="nextTickerHeadline()" title="Jump to Next Headline" aria-label="Next breaking headline" class="ticker-ctrl-btn">
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
        </button>
      </div>
    </div>
  </div>"""


def build_newzire_frontpage_bands(articles, config, resolver):
    """Renders all 5 interactive Newzire homepage bands populated from articles_db.json."""
    used_hero_slugs = set()

    # --- Select Lead Story ---
    lead_art = next((a for a in articles if a.get("is_lead")), None) or articles[0]
    lead_slug = clean_slug(lead_art.get("slug", "lead"))
    used_hero_slugs.add(lead_slug)

    # --- Select Editor's Picks (Left Rail: 2 Cards + 1 Spotlight) ---
    picks_candidates = [
        a for a in articles
        if clean_slug(a.get("slug", "")) not in used_hero_slugs and (a.get("editors_pick") or a.get("manual"))
    ]
    for a in articles:
        if len(picks_candidates) >= 2:
            break
        s = clean_slug(a.get("slug", ""))
        if s not in used_hero_slugs and a not in picks_candidates:
            picks_candidates.append(a)
    picks = picks_candidates[:2]
    for a in picks:
        used_hero_slugs.add(clean_slug(a.get("slug", "")))

    spotlight_art = next(
        (a for a in articles if clean_slug(a.get("slug", "")) not in used_hero_slugs and a.get("spotlight")),
        None,
    ) or next(
        (a for a in articles if clean_slug(a.get("slug", "")) not in used_hero_slugs and _article_matches_category(a, "opinion", config)),
        None,
    ) or next(
        (a for a in articles if clean_slug(a.get("slug", "")) not in used_hero_slugs),
        lead_art,
    )
    used_hero_slugs.add(clean_slug(spotlight_art.get("slug", "")))

    # --- Select 2 Sub-Highlights under Center Lead Story ---
    sub_highlights = []
    for preferred_cat in ("world", "sports", "india", "technology"):
        if len(sub_highlights) >= 2:
            break
        cand = next(
            (a for a in articles if clean_slug(a.get("slug", "")) not in used_hero_slugs and _article_matches_category(a, preferred_cat, config)),
            None,
        )
        if cand:
            sub_highlights.append(cand)
            used_hero_slugs.add(clean_slug(cand.get("slug", "")))
    for a in articles:
        if len(sub_highlights) >= 2:
            break
        s = clean_slug(a.get("slug", ""))
        if s not in used_hero_slugs:
            sub_highlights.append(a)
            used_hero_slugs.add(s)

    # --- Right Rail: Latest (5 stories) vs Trending (5 stories) ---
    latest_feed = [a for a in articles if clean_slug(a.get("slug", "")) != lead_slug][:5]
    trending_pool = [lead_art] + [
        a for a in articles
        if clean_slug(a.get("slug", "")) != lead_slug and (
            _article_matches_category(a, "technology", config)
            or _article_matches_category(a, "world", config)
            or _article_matches_category(a, "economy", config)
            or _article_matches_category(a, "sports", config)
        )
    ]
    trending_seen = set()
    trending_feed = []
    for a in trending_pool + articles:
        s = clean_slug(a.get("slug", ""))
        if s not in trending_seen:
            trending_seen.add(s)
            trending_feed.append(a)
        if len(trending_feed) >= 5:
            break

    # Resolve LCP Lead Image
    lead_img_src, lead_img_fb = resolver.resolve(lead_art.get("image_url", ""), width=960)
    lead_img_tag = build_img_tag(
        lead_img_src,
        lead_art.get("title", ""),
        width=800,
        height=500,
        fallback_src=lead_img_fb,
        extra_attrs='fetchpriority="high" decoding="sync"',
    )

    # Build Left Rail HTML
    left_pick_blocks = []
    for idx, art in enumerate(picks):
        slug = clean_slug(art.get("slug", ""))
        img_s, img_fb = resolver.resolve(art.get("image_url", ""), width=480)
        img_tag = build_img_tag(img_s, art.get("title", ""), width=400, height=250, fallback_src=img_fb, extra_attrs='loading="eager" decoding="async"')
        byline = html.escape(str(art.get("byline") or art.get("source_name") or "Newzire Desk"), quote=True)
        left_pick_blocks.append(
            f"""            <article class="editorial-card">
              <a href="news/{slug}.html">
                <div class="editorial-thumb-wrap">{img_tag}</div>
                <span class="cat-label">{html.escape(str(art.get('category', 'News')), quote=True)}</span>
                <h3 class="pick-headline">{html.escape(str(art.get('title', '')), quote=True)}</h3>
                <div class="meta-row">
                  <span>{byline}</span>
                  <span>{_estimate_read_minutes(art)} min read</span>
                </div>
              </a>
            </article>"""
        )
        left_pick_blocks.append('            <div class="hairline-divider"></div>')

    spot_slug = clean_slug(spotlight_art.get("slug", ""))
    spot_excerpt = html.escape(_extract_article_excerpt(spotlight_art, 130), quote=True)
    spot_label = html.escape(str(spotlight_art.get("sub_beat") or "Special Analysis"), quote=True)

    # Build Sub-Highlights HTML
    sub_hi_blocks = []
    for art in sub_highlights:
        slug = clean_slug(art.get("slug", ""))
        sub_hi_blocks.append(
            f"""              <a href="news/{slug}.html" class="sub-highlight-card">
                <span class="cat-label">{html.escape(str(art.get('category', 'News')), quote=True)}</span>
                <h3 class="sub-highlight-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
              </a>"""
        )

    # Build Right Rail Latest & Trending Panels
    def _render_rail_items(items, is_trending=False):
        rows = []
        for idx, art in enumerate(items, 1):
            slug = clean_slug(art.get("slug", ""))
            cat = html.escape(str(art.get("category", "News")), quote=True)
            kicker = f"{cat} • #{idx} Trending" if is_trending else cat
            right_meta = f"{_estimate_read_minutes(art)} min read" if is_trending else html.escape(str(art.get("date", "")), quote=True)
            rows.append(
                f"""              <a href="news/{slug}.html" class="rail-feed-item">
                <div class="rail-feed-meta">
                  <span class="cat-label">{kicker}</span>
                  <span style="color:var(--text-muted);">{right_meta}</span>
                </div>
                <h3 class="rail-feed-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
              </a>"""
            )
        return "\n".join(rows)

    lead_cat = html.escape(str(lead_art.get("category", "Politics")), quote=True)
    lead_byline = html.escape(str(lead_art.get("byline") or lead_art.get("source_name") or "Newzire Desk"), quote=True)
    lead_date = html.escape(str(lead_art.get("date", "")), quote=True)
    lead_credit = html.escape(str(lead_art.get("image_courtesy", "Newzire Editorial File")), quote=True)
    lead_bureau = html.escape(str(lead_art.get("bureau") or f"{lead_art.get('source_name', 'Newzire')} • Lead Report"), quote=True)
    lead_excerpt = html.escape(_extract_article_excerpt(lead_art, 230), quote=True)

    band1_html = f"""    <!-- BAND 1: 3-COLUMN HERO GRID (25% | 50% | 25%) -->
    <section id="hero-section" class="site-wrap hero-section">
      <div class="hero-frame">
        <div class="hero-grid">
          <aside class="hero-col-left">
            <div class="rail-header">
              <h2 class="rail-title">Editor's Picks</h2>
              <span class="rail-kicker">Curated</span>
            </div>
{chr(10).join(left_pick_blocks)}
            <a href="news/{spot_slug}.html" class="spotlight-card">
              <div class="spotlight-top">
                <span class="spotlight-badge">★ {spot_label}</span>
                <span class="rail-kicker">Special</span>
              </div>
              <h3 class="spotlight-title">{html.escape(str(spotlight_art.get('title', '')), quote=True)}</h3>
              <p class="spotlight-excerpt">{spot_excerpt}</p>
            </a>
          </aside>

          <div class="hero-col-center">
            <article class="lead-story-card editorial-card">
              <a href="news/{lead_slug}.html">
                <div class="lead-kicker-row">
                  <span class="lead-kicker-dash"></span>
                  <span class="lead-kicker-cat">{lead_cat}</span>
                  <span style="color:var(--text-muted);font-size:12px;">•</span>
                  <span class="lead-kicker-tag">Top Headline</span>
                </div>
                <h1 class="lead-headline">{html.escape(str(lead_art.get('title', '')), quote=True)}</h1>
                <div class="lead-meta-strip">
                  <span class="lead-byline">{lead_byline}</span>
                  <span>{_estimate_read_minutes(lead_art)} min read • {lead_date}</span>
                </div>
                <div class="lead-media-wrap">
                  {lead_img_tag}
                  <div class="lead-media-overlay">
                    <span class="lead-media-caption">{lead_credit}</span>
                    <span class="lead-media-pill">Lead Story</span>
                  </div>
                </div>
                <p class="lead-excerpt">{lead_excerpt}</p>
                <div class="lead-footer-bar">
                  <span class="lead-bureau">{lead_bureau}</span>
                  <span class="lead-cta"><span>Read Full Story</span><span>→</span></span>
                </div>
              </a>
            </article>

            <div class="sub-highlights-grid">
{chr(10).join(sub_hi_blocks)}
            </div>
          </div>

          <aside id="latest-trending-box" class="hero-col-right">
            <div class="right-rail-top">
              <div class="right-rail-heading">
                <span class="pulse-dot"></span>
                <h2 class="rail-title">Latest Wire</h2>
              </div>
              <div class="tab-switch-group" role="tablist" aria-label="Wire Feed">
                <button type="button" role="tab" data-tab-btn="rail" data-tab-key="latest" onclick="switchTabGroup('rail','latest')" class="tab-switch-btn is-active">Latest</button>
                <button type="button" role="tab" data-tab-btn="rail" data-tab-key="trending" onclick="switchTabGroup('rail','trending')" class="tab-switch-btn">Trending</button>
              </div>
            </div>

            <div data-tab-panel="rail" data-tab-key="latest" class="rail-feed-list">
{_render_rail_items(latest_feed, is_trending=False)}
            </div>
            <div data-tab-panel="rail" data-tab-key="trending" class="rail-feed-list hidden">
{_render_rail_items(trending_feed, is_trending=True)}
            </div>

            <a href="india.html" class="rail-all-btn">
              <span>Explore All National &amp; Global News</span>
              <span>→</span>
            </a>
          </aside>
        </div>
      </div>
    </section>"""

    # --- BAND 2: 2x2 INDIA & WORLD COVERAGE GRID ---
    geo_specs = [
        ("block-india", "India", "NATIONAL DESK", ["india"], "india.html"),
        ("block-world", "World", "GLOBAL • DIPLOMACY", ["world"], "world.html"),
        ("block-politics", "Politics & Policy", "GOVERNANCE • PARLIAMENT", ["politics"], "politics.html"),
        ("block-tech-econ", "Tech & Economy", "AI • MARKETS • INNOVATION", ["economy", "technology"], "economy.html"),
    ]
    geo_cards = []
    for block_id, title, subtitle, cat_keys, view_all_href in geo_specs:
        block_arts = _filter_by_categories(articles, cat_keys, config, limit=4)
        if len(block_arts) < 4:
            have = {clean_slug(a.get("slug", "")) for a in block_arts}
            for extra in articles:
                if clean_slug(extra.get("slug", "")) not in have:
                    block_arts.append(extra)
                    have.add(clean_slug(extra.get("slug", "")))
                if len(block_arts) >= 4:
                    break
        b_lead = block_arts[0]
        b_lead_slug = clean_slug(b_lead.get("slug", ""))
        b_img_s, b_img_fb = resolver.resolve(b_lead.get("image_url", ""), width=520)
        b_img_tag = build_img_tag(b_img_s, b_lead.get("title", ""), width=400, height=250, fallback_src=b_img_fb, extra_attrs='loading="lazy" decoding="async"')

        bullets_html = []
        for b_art in block_arts[1:4]:
            bs = clean_slug(b_art.get("slug", ""))
            bullets_html.append(
                f"""            <a href="news/{bs}.html" class="geo-bullet-link">
              <span class="geo-bullet-dot"></span>
              <span>{html.escape(str(b_art.get('title', '')), quote=True)}</span>
            </a>"""
            )

        geo_cards.append(
            f"""        <section id="{block_id}" class="geo-card editorial-card">
          <div class="geo-card-top">
            <div>
              <span class="geo-card-subtitle">{subtitle}</span>
              <h3 class="geo-card-title">{title}</h3>
            </div>
            <a href="{view_all_href}" class="geo-view-all">View All →</a>
          </div>
          <a href="news/{b_lead_slug}.html" class="geo-lead">
            <div class="geo-lead-thumb-wrap">{b_img_tag}</div>
            <div>
              <h4 class="geo-lead-title">{html.escape(str(b_lead.get('title', '')), quote=True)}</h4>
              <p class="geo-lead-meta">{html.escape(str(b_lead.get('date', '')), quote=True)} • {_estimate_read_minutes(b_lead)} min read</p>
            </div>
          </a>
          <div class="geo-bullets">
{chr(10).join(bullets_html)}
          </div>
        </section>"""
        )

    band2_html = f"""    <!-- BAND 2: 2x2 INDIA & WORLD COVERAGE GRID -->
    <section id="regions-section" class="site-wrap band-section">
      <div class="section-header-bar">
        <div>
          <span class="section-kicker">Core Coverage</span>
          <h2 class="section-heading">From India to the World</h2>
        </div>
        <span class="section-sub-note">National • Global • Policy • Markets &amp; AI</span>
      </div>
      <div class="geo-grid">
{chr(10).join(geo_cards)}
      </div>
    </section>"""

    # --- BAND 3: CONFIGURABLE FEATURED PARTNER / DEAL OF THE DAY ---
    promo = config.get("promo_banner", DEFAULT_SITE_CONFIG["promo_banner"])
    band3_html = ""
    if promo.get("enabled", True):
        band3_html = f"""    <!-- BAND 3: CONFIGURABLE FEATURED PARTNER / DEAL OF THE DAY -->
    <section id="deal-section" class="site-wrap band-section">
      <div class="promo-card">
        <div class="promo-accent-bar"></div>
        <div class="promo-inner">
          <div class="promo-left">
            <div class="promo-icon-box" aria-hidden="true">
              <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/>
              </svg>
            </div>
            <div>
              <div class="promo-tags">
                <span class="promo-pill">{html.escape(str(promo.get('badge', 'SPONSORED • DEAL OF THE DAY')), quote=True)}</span>
                <span class="promo-highlight">{html.escape(str(promo.get('highlight', '')), quote=True)}</span>
              </div>
              <h3 class="promo-headline">{html.escape(str(promo.get('headline', '')), quote=True)}</h3>
            </div>
          </div>
          <button type="button" onclick="openDealModal()" class="promo-cta-btn">
            {html.escape(str(promo.get('cta_label', 'View Offer →')), quote=True)}
          </button>
        </div>
      </div>
    </section>"""

    # --- BAND 4: VISUAL STORIES (VIDEOS | PICTURES) & MARKETS / EXPLAINERS DESK ---
    video_arts = [a for a in articles if a.get("media_type") == "video"][:3]
    for a in articles:
        if len(video_arts) >= 3:
            break
        if a not in video_arts:
            video_arts.append(a)

    photo_arts = [a for a in articles if a.get("media_type") == "photo"][:3]
    for a in articles:
        if len(photo_arts) >= 3:
            break
        if a not in photo_arts and a not in video_arts:
            photo_arts.append(a)

    def _render_media_cards(items, is_video=True):
        cards = []
        default_badges = ("05:40", "08:15", "04:25") if is_video else ("14 Photos", "11 Photos", "16 Photos")
        default_tag = "Briefing" if is_video else "Photo Essay"
        for idx, art in enumerate(items[:3]):
            slug = clean_slug(art.get("slug", ""))
            img_s, img_fb = resolver.resolve(art.get("image_url", ""), width=480)
            img_tag = build_img_tag(img_s, art.get("title", ""), width=400, height=250, fallback_src=img_fb, extra_attrs='loading="lazy" decoding="async"')
            badge = html.escape(str(art.get("media_badge") or default_badges[idx % 3]), quote=True)
            tag = html.escape(str(art.get("media_tag") or default_tag), quote=True)
            play_html = (
                '<span class="media-play-badge" aria-hidden="true">'
                '<svg width="13" height="13" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>'
                '</span>'
                if is_video
                else ""
            )
            cards.append(
                f"""              <a href="news/{slug}.html" class="media-card editorial-card">
                <div class="media-card-thumb-wrap">
                  {img_tag}
                  {play_html}
                  <span class="media-card-badge">{badge}</span>
                </div>
                <div class="media-card-body">
                  <span class="cat-label">{tag}</span>
                  <h3 class="media-card-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
                </div>
              </a>"""
            )
        return "\n".join(cards)

    markets_arts = _filter_by_categories(articles, ["economy", "technology", "opinion"], config, limit=3)
    markets_items_html = []
    for art in markets_arts:
        slug = clean_slug(art.get("slug", ""))
        sub_label = html.escape(str(art.get("sub_beat") or art.get("category", "Markets")), quote=True)
        markets_items_html.append(
            f"""              <a href="news/{slug}.html" class="desk-list-item">
                <span class="desk-item-meta">{sub_label} • {_estimate_read_minutes(art)} min read</span>
                <h3 class="desk-item-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
              </a>"""
        )

    band4_html = f"""    <!-- BAND 4: MULTIMEDIA (VIDEOS / PICTURES) & MARKETS / EXPLAINERS DESK -->
    <section id="visual-markets-section" class="band-alt-section">
      <div class="site-wrap">
        <div class="band-4-grid">
          <div id="media-desk-box" class="media-desk-col">
            <div class="media-desk-header">
              <div>
                <span class="section-kicker">Multimedia Desk</span>
                <h2 class="section-heading" style="font-size:21px;">Videos &amp; Photo Essays</h2>
              </div>
              <div class="media-tabs" role="tablist" aria-label="Multimedia Format">
                <button type="button" role="tab" data-tab-btn="media" data-tab-key="videos" onclick="switchTabGroup('media','videos')" class="media-tab-btn is-active">
                  <svg width="13" height="13" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
                  <span>Videos</span>
                </button>
                <button type="button" role="tab" data-tab-btn="media" data-tab-key="pictures" onclick="switchTabGroup('media','pictures')" class="media-tab-btn">
                  <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
                  <span>Pictures</span>
                </button>
              </div>
            </div>

            <div data-tab-panel="media" data-tab-key="videos" class="media-cards-grid">
{_render_media_cards(video_arts, is_video=True)}
            </div>
            <div data-tab-panel="media" data-tab-key="pictures" class="media-cards-grid hidden">
{_render_media_cards(photo_arts, is_video=False)}
            </div>
          </div>

          <aside id="markets-desk-box" class="markets-desk-col">
            <div class="explainers-top-stripe"></div>
            <div class="markets-desk-inner">
              <div class="markets-desk-head">
                <div>
                  <span class="section-kicker" style="color:var(--amber);">Strategic Briefing</span>
                  <h2 class="panel-title">Markets &amp; Explainers</h2>
                </div>
                <span class="markets-pill">Deep Dive</span>
              </div>
              <div class="desk-list">
{chr(10).join(markets_items_html)}
              </div>
              <a href="economy.html" class="desk-footer-btn">Explore Economy &amp; Analysis →</a>
            </div>
          </aside>
        </div>
      </div>
    </section>"""

    # --- BAND 5: 3-COLUMN VIEWS, CULTURE & READER SERVICES ---
    views_arts = _filter_by_categories(articles, ["opinion", "politics", "world"], config, limit=4)
    views_items_html = []
    for art in views_arts:
        slug = clean_slug(art.get("slug", ""))
        tag = html.escape(str(art.get("sub_beat") or art.get("category", "Opinion")), quote=True)
        author = html.escape(str(art.get("byline") or art.get("source_name") or "Newzire Desk"), quote=True)
        views_items_html.append(
            f"""            <a href="news/{slug}.html" class="views-item">
              <div class="views-meta">
                <span class="views-tag">{tag}</span>
                <span style="color:var(--text-muted);">•</span>
                <span class="views-author">{author}</span>
              </div>
              <h3 class="views-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
            </a>"""
        )

    culture_all = _filter_by_categories(articles, ["culture"], config, limit=6)
    arts_list = [a for a in culture_all if a.get("sub_beat") == "arts"] + [
        a for a in articles if _article_matches_category(a, "sports", config) or _article_matches_category(a, "culture", config)
    ]
    living_list = [a for a in culture_all if a.get("sub_beat") == "living"] + [
        a for a in articles if _article_matches_category(a, "india", config) or _article_matches_category(a, "economy", config)
    ]
    heritage_list = [a for a in culture_all if a.get("sub_beat") == "spirit"] + [
        a for a in articles if _article_matches_category(a, "opinion", config) or _article_matches_category(a, "culture", config)
    ]

    def _dedupe_slice(seq, n=3):
        out, seen = [], set()
        for item in seq:
            s = clean_slug(item.get("slug", ""))
            if s not in seen:
                seen.add(s)
                out.append(item)
            if len(out) >= n:
                break
        return out

    def _render_culture_cards(items, fallback_tag):
        out = []
        for art in _dedupe_slice(items, 3):
            slug = clean_slug(art.get("slug", ""))
            cat = html.escape(str(art.get("category") or fallback_tag), quote=True)
            out.append(
                f"""            <a href="news/{slug}.html" class="culture-mini-card">
              <span class="cat-label">{cat} • {_estimate_read_minutes(art)} min read</span>
              <h3 class="culture-mini-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
            </a>"""
            )
        return "\n".join(out)

    band5_html = f"""    <!-- BAND 5: MAGAZINE, CULTURE & READER SERVICES ROW -->
    <section id="magazine-services-section" class="site-wrap band-5-section">
      <div class="band-5-grid">
        <div id="views-column" class="band-5-col">
          <div class="surface-panel">
            <div class="panel-header">
              <div>
                <span class="section-kicker" style="font-size:10px;">Views &amp; Specials</span>
                <h2 class="panel-title">Views &amp; Analysis</h2>
              </div>
              <a href="opinion.html" class="geo-view-all">All →</a>
            </div>
            <div style="flex:1;">
{chr(10).join(views_items_html)}
            </div>
          </div>
        </div>

        <div id="culture-column" class="band-5-col">
          <div class="surface-panel">
            <div class="panel-header" style="margin-bottom:14px;">
              <div>
                <span class="section-kicker" style="font-size:10px;">Culture &amp; Lifestyle</span>
                <h2 class="panel-title">Culture &amp; Life</h2>
              </div>
              <a href="culture.html" class="geo-view-all">All →</a>
            </div>

            <div class="culture-tabs-bar" role="tablist" aria-label="Culture Categories">
              <button type="button" role="tab" data-tab-btn="culture" data-tab-key="arts" onclick="switchTabGroup('culture','arts')" class="culture-tab-btn is-active">Arts &amp; Film</button>
              <button type="button" role="tab" data-tab-btn="culture" data-tab-key="living" onclick="switchTabGroup('culture','living')" class="culture-tab-btn">Living &amp; Food</button>
              <button type="button" role="tab" data-tab-btn="culture" data-tab-key="heritage" onclick="switchTabGroup('culture','heritage')" class="culture-tab-btn">Heritage &amp; Ideas</button>
            </div>

            <div data-tab-panel="culture" data-tab-key="arts" class="culture-cards-list">
{_render_culture_cards(arts_list, "Arts & Film")}
            </div>
            <div data-tab-panel="culture" data-tab-key="living" class="culture-cards-list hidden">
{_render_culture_cards(living_list, "Living")}
            </div>
            <div data-tab-panel="culture" data-tab-key="heritage" class="culture-cards-list hidden">
{_render_culture_cards(heritage_list, "Heritage")}
            </div>
          </div>
        </div>

        <div id="services-column" class="band-5-col services-stack">
          <div class="surface-panel" style="height:auto;">
            <div class="panel-header">
              <div>
                <span class="section-kicker" style="font-size:10px;">Reader Services</span>
                <h2 class="panel-title">Reader Utility Desks</h2>
              </div>
            </div>
            <div class="services-pill-grid">
              <button type="button" onclick="openServiceModal('health-wellness')" class="service-pill-btn"><span class="service-pill-icon">✚</span><span class="service-pill-label">Health &amp; Wellness</span></button>
              <button type="button" onclick="openServiceModal('legal-consumer')" class="service-pill-btn"><span class="service-pill-icon">⚖</span><span class="service-pill-label">Legal &amp; Consumer</span></button>
              <button type="button" onclick="openServiceModal('career-exams')" class="service-pill-btn"><span class="service-pill-icon">✎</span><span class="service-pill-label">Career &amp; Exams</span></button>
              <button type="button" onclick="openServiceModal('jobs-desk')" class="service-pill-btn"><span class="service-pill-icon">◈</span><span class="service-pill-label">Jobs Desk</span></button>
              <button type="button" onclick="openServiceModal('events-calendar')" class="service-pill-btn"><span class="service-pill-icon">◎</span><span class="service-pill-label">Events Calendar</span></button>
              <button type="button" onclick="openServiceModal('civic-directory')" class="service-pill-btn"><span class="service-pill-icon">☎</span><span class="service-pill-label">Civic Directory</span></button>
              <button type="button" onclick="openServiceModal('fact-check')" class="service-pill-btn"><span class="service-pill-icon">✓</span><span class="service-pill-label">Fact-Check Desk</span></button>
              <button type="button" onclick="openServiceModal('archives')" class="service-pill-btn"><span class="service-pill-icon">★</span><span class="service-pill-label">News Archives</span></button>
              <a href="editorial-policy.html" class="service-pill-btn span-2">
                <span style="display:inline-flex;align-items:center;gap:8px;">
                  <span class="service-pill-icon">❖</span>
                  <span class="service-pill-label">Newzire 100% Fact-Lock Editorial Standards</span>
                </span>
                <span style="font-size:12px;color:var(--text-secondary);">→</span>
              </a>
            </div>
          </div>

          <div class="surface-panel" style="height:auto;">
            <div class="tip-box-head">
              <h3 class="tip-box-title">Send a News Tip</h3>
              <span class="tip-box-badge">Reader Desk</span>
            </div>
            <p class="tip-box-desc">Share a verified tip, public-interest story lead, or feedback directly with Newzire editors.</p>
            <form id="reader-tip-form" onsubmit="handleTipSubmit(event)" class="tip-form">
              <div class="tip-row-2">
                <input id="tip-name" type="text" required placeholder="Your Name *" class="form-input">
                <select id="tip-topic" aria-label="Tip Category" class="form-input">
                  <option value="News Tip">News Tip</option>
                  <option value="Correction">Correction</option>
                  <option value="Feedback">Feedback</option>
                  <option value="Press Release">Press Release</option>
                </select>
              </div>
              <textarea id="tip-message" rows="2" required placeholder="Write your message or story details..." class="form-input"></textarea>
              <button type="submit" class="btn-primary-full">Send to Newsroom →</button>
            </form>
            <div id="tip-success-state" class="tip-success-box hidden">
              <strong>✓</strong>
              <div>
                <p style="font-weight:700;">Message received by the Newzire desk!</p>
                <p style="font-size:11px;opacity:0.9;margin-top:2px;">Our editorial team reviews every reader submission.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>"""

    return "\n\n".join([band1_html, band2_html, band3_html, band4_html, band5_html]), lead_img_src


# ==========================================================================
# SUB-PAGE BUILDERS (ARTICLES, CATEGORIES, CUSTOM PAGES, 404, SEO)
# ==========================================================================

def build_individual_article_page(art, all_articles, today_str, config, image_resolver=None):
    """Creates a dedicated SEO-optimized HTML page for a single article inside /news/."""
    os.makedirs(NEWS_DIR, exist_ok=True)
    site_url = get_site_url(config)
    slug = clean_slug(art.get("slug", "news-story"))
    canonical_url = f"{site_url}/news/{slug}.html"

    raw_body_html = str(art.get("content_html", "") or "")
    normalized_body = re.sub(r"<h3\b[^>]*>", "<h2>", raw_body_html, flags=re.IGNORECASE)
    normalized_body = re.sub(r"</h3\s*>", "</h2>", normalized_body, flags=re.IGNORECASE)
    body_html = _sanitize_article_html(normalized_body)
    plain_text = BeautifulSoup(body_html, "html.parser").get_text(" ", strip=True)
    meta_desc_raw = (plain_text[:155].rsplit(" ", 1)[0] + "...") if len(plain_text) > 155 else plain_text
    meta_desc = html.escape(meta_desc_raw.replace('"', "'"), quote=True)

    safe_title_text = str(art.get("title", "") or "")
    safe_title_escaped = html.escape(safe_title_text, quote=True)
    raw_category = str(art.get("category", "India") or "India")
    safe_category = html.escape(raw_category, quote=True)
    cat_href = _category_href(raw_category, config, prefix="../")
    safe_date = html.escape(str(art.get("date", "") or ""), quote=True)
    safe_byline = html.escape(str(art.get("byline") or "Newzire News Desk"), quote=True)
    safe_courtesy = html.escape(str(art.get("image_courtesy", "") or "Photo Credit: Newzire News Desk"), quote=True)
    read_mins = _estimate_read_minutes(art)

    resolver = image_resolver or ImageResolver(skip_check=should_skip_image_check())
    raw_img = str(art.get("image_url", "") or "").strip() or DEFAULT_FALLBACK_IMAGE
    article_img_src, article_img_fallback = resolver.resolve(raw_img, width=960, local_prefix="../")
    og_img_url = f"{site_url}/{raw_img.lstrip('./')}" if not raw_img.startswith(("http://", "https://")) else raw_img
    safe_og_img_url = html.escape(og_img_url, quote=True)
    article_img_tag = build_img_tag(
        article_img_src,
        safe_title_text,
        width=800,
        height=450,
        fallback_src=article_img_fallback,
        extra_attrs='fetchpriority="high" decoding="sync"',
        css_class="",
    )

    schema_ld = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": safe_title_text,
        "image": [og_img_url],
        "datePublished": art.get("iso_date", datetime.now(timezone.utc).isoformat()),
        "dateModified": art.get("iso_date", datetime.now(timezone.utc).isoformat()),
        "author": [{"@type": "Organization", "name": "Newzire News Desk", "url": site_url}],
        "publisher": {"@type": "Organization", "name": SITE_NAME, "url": site_url},
        "description": meta_desc_raw,
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical_url},
    }
    schema_json = json.dumps(schema_ld, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")

    encoded_url = quote(canonical_url, safe="")
    encoded_title = quote(safe_title_text, safe="")

    # Sidebar: Related in Category + Latest Wire
    related = [
        a for a in all_articles
        if clean_slug(a.get("slug", "")) != slug and _article_matches_category(a, raw_category, config)
    ][:4]
    if len(related) < 4:
        seen_rel = {slug} | {clean_slug(a.get("slug", "")) for a in related}
        for a in all_articles:
            s = clean_slug(a.get("slug", ""))
            if s not in seen_rel:
                related.append(a)
                seen_rel.add(s)
            if len(related) >= 4:
                break

    related_html = "\n".join(
        f"""            <a href="{clean_slug(a.get('slug', ''))}.html" class="rail-feed-item">
              <div class="rail-feed-meta">
                <span class="cat-label">{html.escape(str(a.get('category', 'News')), quote=True)}</span>
                <span style="color:var(--text-muted);">{_estimate_read_minutes(a)} min read</span>
              </div>
              <h3 class="rail-feed-title">{html.escape(str(a.get('title', '')), quote=True)}</h3>
            </a>"""
        for a in related
    )

    article_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title_escaped} | {SITE_NAME}</title>
    <meta name="description" content="{meta_desc}">
    <link rel="canonical" href="{canonical_url}">
    <meta property="og:type" content="article">
    <meta property="og:title" content="{safe_title_escaped}">
    <meta property="og:description" content="{meta_desc}">
    <meta property="og:image" content="{safe_og_img_url}">
    <meta property="og:url" content="{canonical_url}">
    <meta property="og:site_name" content="{SITE_NAME}">
    <meta name="twitter:card" content="summary_large_image">
    <script type="application/ld+json">{schema_json}</script>
{get_head_links(css_path="../style.css", preload_img=article_img_src, config=config)}
</head>
<body>
{get_top_bar_html(config, today_str, prefix="../")}

{get_header_html(config, prefix="../")}

  <main class="main-content">
    <div class="site-wrap page-shell">
      <div class="article-layout-grid">
        <article class="article-box">
          <a href="../index.html" class="back-btn">← Back to Frontpage</a>
          <div><a href="{cat_href}" class="badge">{safe_category}</a></div>
          <h1 class="single-title">{safe_title_escaped}</h1>
          <div class="single-meta">
            <span style="font-weight:600;color:var(--text-primary);">{safe_byline}</span>
            <span class="meta-dot"></span>
            <span>{safe_date}</span>
            <span class="meta-dot"></span>
            <span>{read_mins} min read</span>
          </div>

          <div class="share-bar">
            <span class="share-label">Share Story</span>
            <div class="share-buttons">
              <a href="https://wa.me/?text={encoded_title}%20{encoded_url}" target="_blank" rel="noopener noreferrer" class="share-btn whatsapp">WhatsApp</a>
              <a href="https://twitter.com/intent/tweet?text={encoded_title}&amp;url={encoded_url}" target="_blank" rel="noopener noreferrer" class="share-btn">X</a>
              <a href="https://www.facebook.com/sharer/sharer.php?u={encoded_url}" target="_blank" rel="noopener noreferrer" class="share-btn facebook">Facebook</a>
              <button type="button" id="copy-link-btn" onclick="copyCurrentPageLink()" class="share-btn copy">Copy Link</button>
            </div>
          </div>

          <figure class="single-figure">
            {article_img_tag}
            <figcaption class="image-credit">{safe_courtesy}</figcaption>
          </figure>

          <div class="article-content">
            {body_html}
          </div>
        </article>

        <aside class="article-sidebar">
          <div class="surface-panel" style="height:auto;">
            <div class="panel-header">
              <div>
                <span class="section-kicker" style="font-size:10px;">Continue Reading</span>
                <h2 class="panel-title">Related Stories</h2>
              </div>
              <a href="{cat_href}" class="geo-view-all">More →</a>
            </div>
            <div class="rail-feed-list">
{related_html}
            </div>
          </div>
        </aside>
      </div>
    </div>
  </main>

{get_footer_html(config, prefix="../")}
{get_mobile_drawer_and_modals_html(config, prefix="../")}
</body>
</html>"""

    filepath = os.path.join(NEWS_DIR, f"{slug}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(article_html)


def build_category_pages(articles, today_str, config, image_resolver=None):
    """Creates a dedicated SEO HTML archive page for every category in site_config.json."""
    resolver = image_resolver or ImageResolver(skip_check=should_skip_image_check())
    site_url = get_site_url(config)
    generated_files = []

    for cat in config.get("categories", []):
        cat_name = cat["name"].strip()
        safe_cat_name = html.escape(cat_name, quote=True)
        cat_file = os.path.basename(cat["file"])
        valid_aliases = {str(a).strip().lower() for a in cat.get("aliases", []) if str(a).strip()} | {
            cat_name.lower(),
            str(cat.get("id", "")).strip().lower(),
        }

        filtered = [a for a in articles if str(a.get("category", "")).strip().lower() in valid_aliases]

        if filtered:
            cards = []
            for idx, art in enumerate(filtered):
                art_slug = clean_slug(art.get("slug", "news-story"))
                card_img, card_fallback = resolver.resolve(art.get("image_url", ""), width=520)
                load_attr = 'fetchpriority="high"' if idx == 0 else 'loading="lazy"'
                card_img_tag = build_img_tag(
                    card_img,
                    art.get("title", ""),
                    width=400,
                    height=250,
                    fallback_src=card_fallback,
                    extra_attrs=f'{load_attr} decoding="async"',
                    css_class="",
                )
                cards.append(
                    f"""        <a href="news/{art_slug}.html" class="archive-card">
          <div class="archive-card-media">{card_img_tag}</div>
          <div class="archive-card-body">
            <span class="cat-label">{html.escape(str(art.get('category', cat_name)), quote=True)}</span>
            <h2 class="archive-card-title">{html.escape(str(art.get('title', '')), quote=True)}</h2>
            <p class="archive-card-excerpt">{html.escape(_extract_article_excerpt(art, 130), quote=True)}</p>
            <div class="archive-card-meta">{html.escape(str(art.get('date', '')), quote=True)} • {_estimate_read_minutes(art)} min read</div>
          </div>
        </a>"""
                )
            cards_html = "\n".join(cards)
        else:
            cards_html = f'<p style="color:var(--text-secondary);padding:20px 0;">New stories in <strong>{safe_cat_name}</strong> will appear here on the next desk update.</p>'

        page_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_cat_name} News | {SITE_NAME}</title>
    <meta name="description" content="Latest {safe_cat_name} news, analysis, and breaking updates from India and the world on {SITE_NAME}.">
    <link rel="canonical" href="{site_url}/{cat_file}">
{get_head_links(css_path="style.css", config=config)}
</head>
<body>
{get_top_bar_html(config, today_str, prefix="")}

{get_header_html(config, prefix="", active_file=cat_file)}

  <main class="main-content">
    <div class="site-wrap page-shell">
      <div class="category-hero-bar">
        <div>
          <span class="section-kicker">Newzire Desk</span>
          <h1 class="category-page-title">{safe_cat_name}</h1>
        </div>
        <span class="rail-kicker">{len(filtered)} Stories</span>
      </div>
      <div class="archive-grid">
{cards_html}
      </div>
    </div>
  </main>

{get_footer_html(config, prefix="", active_file=cat_file)}
{get_mobile_drawer_and_modals_html(config, prefix="")}
</body>
</html>"""

        with open(cat_file, "w", encoding="utf-8") as f:
            f.write(page_html)
        generated_files.append(cat_file)

    print(f"✅ Generated {len(generated_files)} category HTML pages!")
    return generated_files


def build_custom_pages(today_str, config):
    """Wraps fragments in pages/*.html with live Newzire chrome."""
    site_url = get_site_url(config)
    generated_files = []
    for page in config.get("custom_pages", []):
        output_file = os.path.basename(page.get("output_file", ""))
        fragment_path = page.get("source_fragment", "")
        if not output_file or not fragment_path:
            continue

        if not os.path.exists(fragment_path):
            frag_dir = os.path.dirname(fragment_path)
            if frag_dir:
                os.makedirs(frag_dir, exist_ok=True)
            fragment_html = (
                f'<article class="article-box"><a href="index.html" class="back-btn">← Back to Frontpage</a>'
                f'<h1 class="single-title">{html.escape(str(page.get("title", output_file)), quote=True)}</h1>'
                '<div class="article-content"><p>Content coming soon.</p></div></article>'
            )
            with open(fragment_path, "w", encoding="utf-8") as f:
                f.write(fragment_html)

        with open(fragment_path, "r", encoding="utf-8") as f:
            body_fragment = f.read().strip()

        page_title = html.escape(str(page.get("title", f"{output_file} | {SITE_NAME}")), quote=True)
        meta_desc = html.escape(str(page.get("meta_description", f"{SITE_NAME} — India & World Digital Edition")).replace('"', "'"), quote=True)

        full_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <meta name="description" content="{meta_desc}">
    <link rel="canonical" href="{site_url}/{output_file}">
{get_head_links(css_path="style.css", config=config)}
</head>
<body>
{get_top_bar_html(config, today_str, prefix="")}

{get_header_html(config, prefix="", active_file=output_file)}

  <main class="main-content">
    <div class="site-wrap page-shell">
      {body_fragment}
    </div>
  </main>

{get_footer_html(config, prefix="", active_file=output_file)}
{get_mobile_drawer_and_modals_html(config, prefix="")}
</body>
</html>"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        generated_files.append(output_file)

    print(f"✅ Generated {len(generated_files)} custom HTML pages!")
    return generated_files


def build_404_page(articles, today_str, config, image_resolver=None):
    """Generates a branded custom 404.html page with latest stories and category links."""
    resolver = image_resolver or ImageResolver(skip_check=should_skip_image_check())

    cat_pills = " ".join(
        f'<a href="{html.escape(os.path.basename(cat["file"]), quote=True)}" class="error-404-cat-pill">{html.escape(cat["name"], quote=True)}</a>'
        for cat in config.get("categories", [])
        if isinstance(cat, dict) and cat.get("name") and cat.get("file")
    )

    recent_cards = []
    for art in (articles[:6] if articles else []):
        slug = clean_slug(art.get("slug", "news-story"))
        card_img, card_fb = resolver.resolve(art.get("image_url", ""), width=480)
        img_tag = build_img_tag(card_img, art.get("title", ""), width=400, height=250, fallback_src=card_fb, extra_attrs='loading="lazy" decoding="async"', css_class="")
        recent_cards.append(
            f"""        <a href="news/{slug}.html" class="archive-card">
          <div class="archive-card-media">{img_tag}</div>
          <div class="archive-card-body">
            <span class="cat-label">{html.escape(str(art.get('category', 'News')), quote=True)}</span>
            <h3 class="archive-card-title">{html.escape(str(art.get('title', '')), quote=True)}</h3>
            <div class="archive-card-meta">{html.escape(str(art.get('date', '')), quote=True)}</div>
          </div>
        </a>"""
        )

    recent_section = ""
    if recent_cards:
        recent_section = f"""      <div class="section-header-bar">
        <div>
          <span class="section-kicker">Latest Coverage</span>
          <h2 class="section-heading">Top Stories on Newzire</h2>
        </div>
      </div>
      <div class="archive-grid">
{chr(10).join(recent_cards)}
      </div>"""

    page_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found (404) | {SITE_NAME}</title>
    <meta name="description" content="The page you requested could not be found on {SITE_NAME}. Explore the latest India and World headlines.">
    <meta name="robots" content="noindex, follow">
{get_head_links(css_path="style.css", config=config)}
</head>
<body>
{get_top_bar_html(config, today_str, prefix="")}

{get_header_html(config, prefix="")}

  <main class="main-content">
    <div class="site-wrap page-shell">
      <section class="error-404-box" aria-labelledby="error-404-heading">
        <span class="badge">Error 404 • Page Not Found</span>
        <div class="error-404-number" aria-hidden="true">404</div>
        <h1 class="error-404-title" id="error-404-heading">Sorry, this page or story could not be found</h1>
        <p class="error-404-desc">The link may have moved, expired, or been mistyped. Return to the Newzire frontpage or jump straight into one of our news desks below.</p>
        <div class="error-404-actions">
          <a href="index.html" class="btn-accent">Go to Frontpage</a>
          <a href="india.html" class="btn-secondary">Browse India News</a>
        </div>
        <div class="error-404-cats">{cat_pills}</div>
      </section>
{recent_section}
    </div>
  </main>

{get_footer_html(config, prefix="")}
{get_mobile_drawer_and_modals_html(config, prefix="")}
</body>
</html>"""

    with open("404.html", "w", encoding="utf-8") as f:
        f.write(page_html)
    print("✅ Generated custom 404.html page!")
    return "404.html"


def cleanup_retired_root_pages(current_generated_files):
    """Safely removes root .html files that were previously generated by config but later removed."""
    previous_files = set()
    if os.path.exists(GENERATED_MANIFEST):
        try:
            with open(GENERATED_MANIFEST, "r", encoding="utf-8") as f:
                previous_files = set(json.load(f))
        except Exception:
            previous_files = set()

    current_set = set(current_generated_files)
    protected = {"index.html", "404.html"}

    for obsolete in previous_files - current_set - protected:
        safe_name = os.path.basename(obsolete)
        if safe_name.endswith(".html") and os.path.exists(safe_name):
            os.remove(safe_name)
            print(f"🗑️ Removed retired generated page: {safe_name}")

    with open(GENERATED_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(sorted(current_set), f, ensure_ascii=False, indent=2)


def build_seo_files(articles, config):
    """Generates valid sitemap.xml and robots.txt for search engines."""
    site_url = get_site_url(config)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    urls_xml = f"""  <url>
    <loc>{site_url}/</loc>
    <lastmod>{now_iso}</lastmod>
    <changefreq>hourly</changefreq>
    <priority>1.0</priority>
  </url>\n"""

    for cat in config.get("categories", []):
        cat_file = os.path.basename(cat["file"])
        urls_xml += f"""  <url>
    <loc>{site_url}/{cat_file}</loc>
    <lastmod>{now_iso}</lastmod>
    <changefreq>hourly</changefreq>
    <priority>0.9</priority>
  </url>\n"""

    for page in config.get("custom_pages", []):
        if page.get("include_in_sitemap", True):
            out_file = os.path.basename(page["output_file"])
            freq = page.get("changefreq", "monthly")
            urls_xml += f"""  <url>
    <loc>{site_url}/{out_file}</loc>
    <lastmod>{now_iso}</lastmod>
    <changefreq>{freq}</changefreq>
    <priority>0.7</priority>
  </url>\n"""

    for art in articles:
        art_slug = clean_slug(art.get("slug", "news-story"))
        art_iso = art.get("iso_date", now_iso)
        urls_xml += f"""  <url>
    <loc>{site_url}/news/{art_slug}.html</loc>
    <lastmod>{art_iso}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.8</priority>
  </url>\n"""

    sitemap_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls_xml}</urlset>
"""
    with open("sitemap.xml", "w", encoding="utf-8") as f:
        f.write(sitemap_content)

    robots_content = f"""User-agent: *
Allow: /

Sitemap: {site_url}/sitemap.xml
"""
    with open("robots.txt", "w", encoding="utf-8") as f:
        f.write(robots_content)
    print("✅ Generated sitemap.xml and robots.txt for SEO!")


def build_website_html(articles, config=None, skip_image_check=None, image_resolver=None):
    """Generates index.html, all category pages, custom pages, individual article pages, 404, and SEO files."""
    if config is None:
        config = load_site_config()

    today_str = datetime.now().strftime("%d %B %Y")
    if not articles:
        print("⚠️ No articles available — skipping homepage build.")
        return

    site_url = get_site_url(config)
    resolver = image_resolver or ImageResolver(skip_check=should_skip_image_check(skip_image_check))
    resolver.prewarm([a.get("image_url", "") for a in articles])

    os.makedirs(NEWS_DIR, exist_ok=True)
    valid_files = {f"{clean_slug(art.get('slug', 'news-story'))}.html" for art in articles}
    for fname in os.listdir(NEWS_DIR):
        if fname.endswith(".html") and fname not in valid_files:
            os.remove(os.path.join(NEWS_DIR, fname))

    for art in articles:
        build_individual_article_page(art, articles, today_str, config, image_resolver=resolver)

    bands_html, lcp_img = build_newzire_frontpage_bands(articles, config, resolver)
    ticker_html = build_ticker_html(articles)

    html_template = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{SITE_NAME} — Digital Edition • India &amp; World News Portal</title>
    <meta name="description" content="{SITE_NAME} delivers fast, fact-locked English journalism covering India, World affairs, Politics, Economy, Technology, Sports, Opinion, and Culture.">
    <link rel="canonical" href="{site_url}/">
    <meta property="og:type" content="website">
    <meta property="og:title" content="{SITE_NAME} — India &amp; World Digital News Portal">
    <meta property="og:description" content="Sharp, fact-locked English reporting on India, geopolitics, economy, AI &amp; technology, sports, and culture.">
    <meta property="og:url" content="{site_url}/">
    <meta property="og:site_name" content="{SITE_NAME}">
{get_head_links(css_path="style.css", preload_img=lcp_img, config=config)}
</head>
<body>
{get_top_bar_html(config, today_str, prefix="")}

{get_header_html(config, prefix="", active_file="index.html")}

{ticker_html}

  <main id="top" class="main-content">
{bands_html}
  </main>

{get_footer_html(config, prefix="", active_file="index.html")}
{get_mobile_drawer_and_modals_html(config, prefix="")}
</body>
</html>"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)

    cat_files = build_category_pages(articles, today_str, config, image_resolver=resolver)
    custom_files = build_custom_pages(today_str, config)
    build_404_page(articles, today_str, config, image_resolver=resolver)
    cleanup_retired_root_pages(cat_files + custom_files)
    build_seo_files(articles, config)
    print("✅ Live multi-page SEO & PageSpeed-optimized Newzire website generated!")


# ==========================================================================
# AUTOMATED RSS SCANNING & AI REWRITING PIPELINE
# ==========================================================================

def collect_ranked_candidates(existing_links):
    """Pools unread stories from multi-beat RSS feeds and ranks them with Source + Keyword weights."""
    candidates = []
    for feed_info in RSS_FEEDS:
        source_name = feed_info["source"]
        feed_url = feed_info["url"]
        base_weight = feed_info.get("weight", 5.0)

        print(f"📡 Scanning {source_name}...")
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                link = entry.get("link", "")
                if not link or link in existing_links:
                    continue

                title = entry.get("title", "").strip()
                raw_summary = entry.get("summary", "") or entry.get("description", "")
                summary = BeautifulSoup(raw_summary, "html.parser").get_text().strip()
                combined_text = f"{title} {summary}".lower()

                if any(bad in combined_text for bad in BLOCKED_KEYWORDS):
                    continue

                score = base_weight
                for kw, kw_weight in KEYWORD_BIAS.items():
                    if re.search(r"\b" + re.escape(kw.lower()) + r"\b", combined_text):
                        score += kw_weight

                rss_img = get_rss_image(entry)
                if rss_img:
                    score += 1.0

                candidates.append({
                    "title": title,
                    "summary": summary[:700],
                    "link": link,
                    "source_name": source_name,
                    "rss_image_url": rss_img,
                    "score": round(score, 2),
                })
        except Exception as e:
            print(f"⚠️ Failed to fetch {source_name}: {e}")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def fetch_and_rewrite():
    config = load_site_config()
    ai_categories = [c["name"].strip() for c in config.get("categories", []) if c.get("active_for_ai", True)]
    if not ai_categories:
        ai_categories = [c["name"].strip() for c in config.get("categories", [])] or ["India"]
    category_choices_str = ", ".join(f'"{name}"' for name in ai_categories)

    articles = []
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                articles = json.load(f)
        except Exception:
            articles = []

    for a in articles:
        a["slug"] = clean_slug(a.get("slug", "news-story"))
        if "iso_date" not in a:
            a["iso_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    existing_links = {a.get("original_link") for a in articles}
    target_count = 3 if len(articles) == 0 else 1
    added = 0

    for _ in range(target_count):
        candidates = collect_ranked_candidates(existing_links)
        if not candidates:
            print("ℹ️ All matching stories across RSS feeds have already been published.")
            break

        shortlist = candidates[:5]
        candidates_text = ""
        for idx, c in enumerate(shortlist):
            candidates_text += (
                f"\n[{idx}] Source: {c['source_name']} (Score: {c['score']})\n"
                f"Title: {c['title']}\n"
                f"Summary: {c['summary']}\n"
            )

        prompt = f"""
You are a Senior English News Editor at Newzire covering India and World affairs.
Below are {len(shortlist)} news candidates ([0] to [{len(shortlist)-1}]).

1. Story Selection Priorities:
{SELECTION_BIAS_PROMPT}

2. Editorial Framing & Tone:
{TONE_AND_FRAMING_BIAS}

Instructions:
- Step 1: Pick the single highest-impact story using the Selection Priorities and put its integer index in "selected_index".
- Step 2: Write an original, publication-grade 350–450 word English news article with crisp <p> paragraphs and informative <h2> subheadings. Do not copy verbatim from the RSS summary.
- Step 3 (100% Fact Lock): Keep every fact, figure, date, name, and quote 100% faithful to the source. Do not invent facts.
- Step 4 (Photo Strategy): Use "original" when the story features a specific leader, venue, or breaking event; use "stock" for broad technology, AI, economy, or macro trend stories.
- Respond ONLY with a valid JSON object containing these exact 7 keys:
   - "selected_index": integer index of the chosen candidate (e.g., 0 or 1)
   - "title": sharp, authoritative English headline
   - "category": choose exactly ONE from: {category_choices_str}
   - "slug": 4-6 word lowercase hyphenated English SEO slug (e.g., "india-semiconductor-corridor-expansion")
   - "content_html": valid HTML body using only <p> and <h2> tags
   - "image_strategy": "original" or "stock"
   - "stock_search_query": 2-word English stock photo keyword

Candidates:
{candidates_text}
"""

        article_data = generate_with_fallback(prompt)

        chosen_idx = article_data.get("selected_index", 0)
        if isinstance(chosen_idx, bool) or not isinstance(chosen_idx, int) or chosen_idx < 0 or chosen_idx >= len(shortlist):
            chosen_idx = 0
        chosen = shortlist[chosen_idx]

        print(f"🎯 Selected [{chosen_idx}] ({chosen['source_name']} | Score {chosen['score']}): {chosen['title']}")

        strategy = str(article_data.get("image_strategy", "original") or "original").strip().lower()
        search_query = str(article_data.get("stock_search_query", "news") or "news").strip()
        rss_image_url = chosen["rss_image_url"]
        source_name = chosen["source_name"]

        final_image_url, image_courtesy = (None, "")
        if strategy == "stock":
            final_image_url, image_courtesy = fetch_free_stock_image(search_query)
        if not final_image_url and rss_image_url:
            final_image_url = rss_image_url
            image_courtesy = f"Photo Credit: {source_name} (RSS Feed)"
        if not final_image_url:
            final_image_url = DEFAULT_FALLBACK_IMAGE
            image_courtesy = "Photo Credit: Unsplash (Representational)"

        validated_category = normalize_ai_category(
            article_data.get("category"),
            config,
            ai_categories,
        )

        new_article = {
            "title": article_data["title"].strip(),
            "category": validated_category,
            "slug": clean_slug(article_data.get("slug", "newzire-story")),
            "content_html": article_data["content_html"],
            "image_url": final_image_url,
            "image_courtesy": image_courtesy,
            "original_link": chosen["link"],
            "source_name": source_name,
            "byline": f"Newzire {validated_category} Desk",
            "date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "iso_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }

        articles.insert(0, new_article)
        existing_links.add(chosen["link"])
        added += 1
        time.sleep(2)

    if added == 0:
        print("ℹ️ All stories across RSS feeds have already been published. No new articles generated.")

    articles = articles[:100]
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    build_website_html(articles, config)


if __name__ == "__main__":
    if "--rebuild-only" in sys.argv:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                articles = json.load(f)
            for a in articles:
                a["slug"] = clean_slug(a.get("slug", "news-story"))
                if "iso_date" not in a:
                    a["iso_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            build_website_html(articles)
        else:
            print(f"ℹ️ {ARCHIVE_FILE} not found — nothing to rebuild.")
    else:
        fetch_and_rewrite()
