import json
import os
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import news_bot


VALID_EN_PAYLOAD = {
    "selected_index": 0,
    "title": "India Expands Semiconductor Corridors With New Research Alliances",
    "category": "Technology",
    "slug": "india-expands-semiconductor-corridors",
    "content_html": (
        "<p>India has announced an ambitious expansion of its national semiconductor packaging "
        "and design corridors to strengthen domestic electronics manufacturing and global exports.</p>"
        "<h2>Industry Partnerships and Talent Scale</h2>"
        "<p>Universities and private fabrication partners are collaborating to train thousands of "
        "chip engineers across Bengaluru, Gujarat, and Tamil Nadu over the next five years.</p>"
    ),
    "image_strategy": "stock",
    "stock_search_query": "semiconductor chip",
}


class TestModelFallbackChain(unittest.TestCase):
    def test_exact_15_model_priority_order(self):
        expected = [
            ("google", "gemini-3.8-flash"),
            ("google", "gemini-4-hydrogen"),
            ("google", "gemini-3.6-flash"),
            ("google", "gemini-3.5-flash-lite"),
            ("google", "gemini-3.1-flash-lite"),
            ("google", "gemini-3-flash-preview"),
            ("google", "gemma-4-31b-it"),
            ("google", "gemma-4-26b-a4b-it"),
            ("nvidia_nim", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
            ("nvidia_nim", "mistralai/mistral-large-4-1t-a52b"),
            ("nvidia_nim", "nvidia/nemotron-3-ultra-550b-a55b"),
            ("nvidia_nim", "mistralai/mistral-small-4-119b-2603"),
            ("nvidia_nim", "qwen/qwen2.5-72b-instruct"),
            ("nvidia_nim", "openai/gpt-oss-120b"),
            ("nvidia_nim", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        ]
        actual = [(e["provider"], e["model"]) for e in news_bot.MODEL_FALLBACK_CHAIN]
        self.assertEqual(actual, expected)
        self.assertEqual(len(news_bot.MODELS_TO_TRY), 15)

    def test_fails_over_from_google_to_nvidia_nim(self):
        called = []

        def fake_google(model, prompt, client):
            called.append(("google", model))
            raise RuntimeError("429 quota exceeded")

        def fake_nim(model, prompt, key):
            called.append(("nvidia_nim", model))
            return json.dumps(VALID_EN_PAYLOAD)

        with patch.object(news_bot, "_resolve_google_client", return_value=MagicMock()), \
             patch.object(news_bot, "_resolve_nvidia_key", return_value="nvapi-test"), \
             patch.object(news_bot, "_call_google_model", side_effect=fake_google), \
             patch.object(news_bot, "_call_nvidia_nim_model", side_effect=fake_nim), \
             patch.object(news_bot.time, "sleep"):
            result = news_bot.generate_with_fallback("Write a story")

        self.assertEqual(result["title"], VALID_EN_PAYLOAD["title"])
        self.assertEqual(len(called), 9)
        self.assertEqual(called[-1], ("nvidia_nim", "nvidia/nemotron-3.5-super-vl-120b-a12b"))

    def test_auth_401_circuit_breaks_google_provider(self):
        called = []

        def fake_google(model, prompt, client):
            called.append(("google", model))
            raise news_bot._ProviderHttpError("HTTP 401 Unauthorized", status_code=401, is_auth=True)

        def fake_nim(model, prompt, key):
            called.append(("nvidia_nim", model))
            return json.dumps(VALID_EN_PAYLOAD)

        with patch.object(news_bot, "_resolve_google_client", return_value=MagicMock()), \
             patch.object(news_bot, "_resolve_nvidia_key", return_value="nvapi-test"), \
             patch.object(news_bot, "_call_google_model", side_effect=fake_google), \
             patch.object(news_bot, "_call_nvidia_nim_model", side_effect=fake_nim), \
             patch.object(news_bot.time, "sleep"):
            result = news_bot.generate_with_fallback("Write a story")

        self.assertEqual(result["slug"], VALID_EN_PAYLOAD["slug"])
        self.assertEqual(called, [
            ("google", "gemini-3.8-flash"),
            ("nvidia_nim", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
        ])

    def test_nim_http_400_retries_without_response_format(self):
        resp_400 = MagicMock(status_code=400, text="response_format unsupported")
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {
            "choices": [{"message": {"content": json.dumps(VALID_EN_PAYLOAD)}, "finish_reason": "stop"}]
        }
        with patch.object(news_bot.requests, "post", side_effect=[resp_400, resp_200]) as mock_post:
            raw = news_bot._call_nvidia_nim_model("qwen/qwen2.5-72b-instruct", "prompt", "nv-key")
        self.assertIn("Semiconductor", raw)
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("response_format", mock_post.call_args_list[0].kwargs["json"])
        self.assertNotIn("response_format", mock_post.call_args_list[1].kwargs["json"])


class TestEnglishParsingAndValidation(unittest.TestCase):
    def test_parses_think_tags_and_code_fences(self):
        raw = f"<think>Selecting story 0</think>\n```json\n{json.dumps(VALID_EN_PAYLOAD)}\n```"
        parsed = news_bot.parse_ai_json(raw)
        validated = news_bot.validate_article_payload(parsed)
        self.assertEqual(validated["selected_index"], 0)
        self.assertEqual(validated["category"], "Technology")

    def test_rejects_hindi_or_short_payloads(self):
        hindi_payload = {
            "title": "भारत की अंतरिक्ष में बड़ी छलांग",
            "content_html": "<p>इसरो ने नया उपग्रह प्रक्षेपित किया और कई नए रिकॉर्ड बनाए।</p>",
        }
        with self.assertRaises(ValueError):
            news_bot.validate_article_payload(hindi_payload)

        short_payload = {"title": "Short News", "content_html": "<p>Too short.</p>"}
        with self.assertRaises(ValueError):
            news_bot.validate_article_payload(short_payload)

    def test_category_alias_normalization(self):
        cfg = news_bot.DEFAULT_SITE_CONFIG
        ai_cats = [c["name"] for c in cfg["categories"]]
        self.assertEqual(news_bot.normalize_ai_category("Global", cfg, ai_cats), "World")
        self.assertEqual(news_bot.normalize_ai_category("AI", cfg, ai_cats), "Technology")
        self.assertEqual(news_bot.normalize_ai_category("Markets", cfg, ai_cats), "Economy")
        self.assertEqual(news_bot.normalize_ai_category("Unknown", cfg, ai_cats), "India")


class TestImageOptimizationAndSanitization(unittest.TestCase):
    def test_wsrv_optimization_and_fallback_attrs(self):
        resolver = news_bot.ImageResolver(skip_check=True)
        src, fb = resolver.resolve("https://example.com/photo one.jpg", width=640)
        self.assertTrue(src.startswith("https://wsrv.nl/?url="))
        self.assertIn("&w=640&output=webp", src)
        self.assertEqual(fb, "https://example.com/photo%20one.jpg")
        tag = news_bot.build_img_tag(src, 'Quote "Test"', 640, 400, fallback_src=fb)
        self.assertIn('data-fallback-src="https://example.com/photo%20one.jpg"', tag)
        self.assertIn("onerror=", tag)

    def test_xss_html_sanitization(self):
        dirty = (
            '<p>Safe intro <script>alert(1)</script><a href="javascript:evil()">bad</a> '
            '<a href="https://example.com">good</a></p><iframe src="https://evil.com"></iframe>'
        )
        clean = news_bot._sanitize_article_html(dirty)
        self.assertNotIn("<script", clean)
        self.assertNotIn("javascript:", clean)
        self.assertNotIn("<iframe", clean)
        self.assertIn('href="https://example.com"', clean)
        self.assertIn('rel="noopener noreferrer"', clean)


class TestStaticSiteGeneration(unittest.TestCase):
    def test_full_build_website_html_and_integrity(self):
        with open(os.path.join(REPO_ROOT, "articles_db.json"), "r", encoding="utf-8") as f:
            seed_articles = json.load(f)
        with open(os.path.join(REPO_ROOT, "site_config.json"), "r", encoding="utf-8") as f:
            seed_config = json.load(f)

        # Inject synthetic internal CMS metadata & XSS probe into one article copy to verify isolation
        test_articles = json.loads(json.dumps(seed_articles))
        test_articles[0]["author_username"] = "secret.editor@newzire.internal"
        test_articles[0]["content_html"] += '<script>window.__xss=1;</script>'

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                os.makedirs("pages", exist_ok=True)
                for fname in ("about.html", "contact.html", "editorial-policy.html", "privacy-policy.html"):
                    src_p = os.path.join(REPO_ROOT, "pages", fname)
                    with open(src_p, "r", encoding="utf-8") as rf, open(os.path.join("pages", fname), "w", encoding="utf-8") as wf:
                        wf.write(rf.read())

                news_bot.build_website_html(test_articles, config=seed_config, skip_image_check=True)

                # Verify core files exist
                expected_root = [
                    "index.html", "404.html", "sitemap.xml", "robots.txt",
                    "india.html", "world.html", "politics.html", "economy.html",
                    "technology.html", "sports.html", "opinion.html", "culture.html",
                    "about.html", "contact.html", "editorial-policy.html", "privacy-policy.html",
                ]
                for ef in expected_root:
                    self.assertTrue(os.path.exists(ef), f"Missing generated file: {ef}")

                # Check index.html structure
                with open("index.html", "r", encoding="utf-8") as f:
                    idx_html = f.read()

                self.assertIn("Newz", idx_html)
                self.assertIn('href="style.css"', idx_html)
                self.assertNotIn("cdn.tailwindcss.com", idx_html)
                self.assertIn("Editor's Picks", idx_html)
                self.assertIn("From India to the World", idx_html)
                self.assertIn("Videos &amp; Photo Essays", idx_html)
                self.assertIn("Markets &amp; Explainers", idx_html)
                self.assertIn("Views &amp; Analysis", idx_html)
                self.assertIn("Reader Utility Desks", idx_html)
                self.assertIn("Send a News Tip", idx_html)
                self.assertIn("newzire-theme", idx_html)

                # Verify zero Hindi/Devanagari or Tripura/Kokborok/Aabtak24 strings anywhere
                forbidden_re = re.compile(r"[\u0900-\u097F]|tripura|agartala|kokborok|aabtak24", re.IGNORECASE)
                all_html_files = [f for f in expected_root if f.endswith(".html")] + [
                    os.path.join("news", n) for n in os.listdir("news") if n.endswith(".html")
                ]
                self.assertEqual(len(os.listdir("news")), len(test_articles))

                for hf in all_html_files:
                    with open(hf, "r", encoding="utf-8") as rf:
                        content = rf.read()
                    self.assertIsNone(
                        forbidden_re.search(content),
                        f"Forbidden Hindi/regional/old-brand text found in {hf}",
                    )
                    self.assertNotIn("secret.editor@newzire.internal", content)
                    self.assertNotIn("window.__xss", content)

                # Verify valid XML sitemap
                tree = ET.parse("sitemap.xml")
                root = tree.getroot()
                self.assertTrue(root.tag.endswith("urlset"))
                self.assertGreaterEqual(len(list(root)), len(test_articles) + 12)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
