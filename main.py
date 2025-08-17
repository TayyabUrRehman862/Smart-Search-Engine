
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import time
import math
import re
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Config (extends your existing)
# -----------------------------
class SearchEngineConfig:
    """Configuration management for the search engine (CSE-ready)."""
    def __init__(self):
        # Models (keep your defaults)
        self.TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.VISION_MODEL = "openai/clip-vit-base-patch32"

        # System
        self.DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "0") == "1" else "cpu"
        self.MAX_BATCH_SIZE = 32
        self.EMBEDDING_DIM = 384

        # Storage
        self.DATA_DIR = Path("/kaggle/working/data")
        self.MODEL_DIR = Path("/kaggle/working/models")
        self.OUTPUT_DIR = Path("/kaggle/working/output")

        # Collection
        self.SCRAPING_DELAY = 0.3       # delay between API calls (seconds)
        self.MAX_RESULTS_PER_SOURCE = 20
        self.MIN_TEXT_LENGTH = 50

        # NEW: Google CSE credentials (pulled from env by default)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

        for p in [self.DATA_DIR, self.MODEL_DIR, self.OUTPUT_DIR]:
            p.mkdir(parents=True, exist_ok=True)

    def ensure_cse_ready(self):
        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            raise RuntimeError(
                "Google CSE not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
            )

# --------------------------------------------
# Google Web Scraper (via Custom Search API)
# --------------------------------------------
class GoogleWebScraper:
    """Google Custom Search API client with pagination and basic hygiene."""

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, config: SearchEngineConfig, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        self.config = config
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.cse_id = cse_id or config.GOOGLE_CSE_ID
        if not self.api_key or not self.cse_id:
            raise ValueError("GoogleWebScraper requires api_key and cse_id (cx).")

    def search_google(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        """Fetch up to num_results results from Google CSE (max 10 per request)."""
        target = int(num_results or self.config.MAX_RESULTS_PER_SOURCE)
        print(f"🔍 Searching Google CSE for: '{query}' (target {target} results)")

        results: List[Dict] = []
        page_size = 10
        pages = math.ceil(target / page_size)
        start_index = 1  # CSE uses 1-based indexing

        for page in range(pages):
            take = min(page_size, target - len(results))
            if take <= 0:
                break

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": take,
                "start": start_index,
                # optional: "gl": "us", "lr": "lang_en"  # locale/language hints
            }

            try:
                r = requests.get(self.BASE_URL, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()

                # Common API error patterns
                if "error" in data:
                    code = data["error"].get("code")
                    msg = data["error"].get("message", "")
                    reason = ",".join([e.get("reason", "") for e in data["error"].get("errors", [])])
                    raise RuntimeError(f"CSE API error {code}: {msg} ({reason})")

                for item in data.get("items", []):
                    title = item.get("title", "").strip()
                    snippet = item.get("snippet", "").strip()
                    link = item.get("link", "").strip()
                    if not title or not link:
                        continue
                    results.append({
                        "title": title,
                        "url": link,
                        "content": snippet,
                        "source": "google_search",
                        "query": query,
                    })

                # Next start index (CSE ignores if past end)
                start_index += page_size
                if len(results) >= target:
                    break

                time.sleep(self.config.SCRAPING_DELAY)
            except Exception as e:
                print(f"   ❌ Error calling CSE: {e}")
                break

        print(f"   ✅ Retrieved {len(results)} Google results")
        return results

    def get_page_content(self, url: str, max_chars: int = 8000) -> str:
        """Fetch and clean page text (simple BS4; tries readability if installed)."""
        try:
            # Try readability if available for better article extraction
            try:
                from readability import Document  # type: ignore
                use_readability = True
            except Exception:
                use_readability = False

            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            html = resp.text

            if use_readability:
                doc = Document(html)
                content_html = doc.summary()
                soup = BeautifulSoup(content_html, "html.parser")
                text = soup.get_text(" ")
            else:
                soup = BeautifulSoup(html, "html.parser")
                for t in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
                    t.decompose()
                text = soup.get_text(" ")

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            print(f"   ⚠️  Error fetching {url}: {e}")
            return ""

# -------------------------------------------------
# WikipediaDataSource (unchanged API – included for completeness)
# -------------------------------------------------
class WikipediaDataSource:
    def __init__(self, config: SearchEngineConfig):
        self.config = config
        try:
            import wikipedia  # lazy import
            self.wiki = wikipedia
            self.wiki.set_lang("en")
        except Exception:
            self.wiki = None
            print("⚠️  Install wikipedia: pip install wikipedia-api")

    def search_articles(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        if not self.wiki:
            return []
        limit = int(max_results or self.config.MAX_RESULTS_PER_SOURCE)
        print(f"📚 Searching Wikipedia for: '{query}' (max {limit} results)")
        out: List[Dict] = []
        try:
            titles = self.wiki.search(query, results=limit * 2)  # fetch extras, we'll filter
            for t in titles[:limit]:
                try:
                    summary = self.wiki.summary(t, sentences=4, auto_suggest=False)
                    page = self.wiki.page(t, auto_suggest=False)
                    if len(summary) < self.config.MIN_TEXT_LENGTH:
                        continue
                    out.append({
                        "title": t,
                        "content": summary,
                        "url": page.url,
                        "source": "wikipedia",
                        "query": query,
                    })
                    time.sleep(0.05)
                except Exception:
                    continue
            print(f"   ✅ Retrieved {len(out)} Wikipedia articles")
            return out
        except Exception as e:
            print(f"   ❌ Wikipedia error: {e}")
            return []

# ---------------------------------
# GoogleDataAggregator (drop-in)
# ---------------------------------
class GoogleDataAggregator:
    def __init__(self, config: SearchEngineConfig, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        self.config = config
        self.web_scraper = GoogleWebScraper(config, api_key=api_key, cse_id=cse_id)
        self.wikipedia_source = WikipediaDataSource(config)

    def collect_comprehensive_data(self, queries: List[str], sources: Optional[List[str]] = None) -> List[Dict]:
        sources = sources or ["web", "wikipedia"]
        all_docs: List[Dict] = []
        print("\n" + "=" * 60)
        print(f"COMPREHENSIVE DATA COLLECTION — Sources: {sources}")
        print("=" * 60)
        for i, q in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {q}")
            if "web" in sources:
                all_docs.extend(self.web_scraper.search_google(q, self.config.MAX_RESULTS_PER_SOURCE))
            if "wikipedia" in sources:
                all_docs.extend(self.wikipedia_source.search_articles(q, self.config.MAX_RESULTS_PER_SOURCE))
            time.sleep(self.config.SCRAPING_DELAY)
        unique = self._remove_duplicates(all_docs)
        print(f"\n📊 Collection Summary: total={len(all_docs)} unique={len(unique)} dupes_removed={len(all_docs)-len(unique)}")
        return unique

    @staticmethod
    def _remove_duplicates(documents: List[Dict]) -> List[Dict]:
        seen = set()
        uniq: List[Dict] = []
        for d in documents:
            key = (d.get("title", "").lower().strip(), d.get("url", "").lower().strip())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        return uniq

    def prepare_for_indexing(self, documents: List[Dict]) -> Tuple[List[str], List[Dict]]:
        texts: List[str] = []
        meta: List[Dict] = []
        for d in documents:
            title = (d.get("title") or "").strip()
            content = (d.get("content") or "").strip()
            text = f"{title}. {content}" if title and content else content
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < self.config.MIN_TEXT_LENGTH:
                continue
            texts.append(text)
            meta.append({
                "source": d.get("source", "unknown"),
                "url": d.get("url", ""),
                "query": d.get("query", ""),
                "title": title,
                "original_content": content,
            })
        print(f"📝 Prepared {len(texts)} docs for indexing")
        return texts, meta

# =====================================================
# Advanced Gradio UI for Google + Wikipedia Search
# =====================================================
# =====================================================
# Unified Column UI with Loading Button
# =====================================================
import gradio as gr
from transformers import pipeline

# Initialize aggregator
cfg = SearchEngineConfig()
cfg.ensure_cse_ready()
aggregator = GoogleDataAggregator(cfg)

# Load summarizer model (open-source)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def search_and_answer(query, sources):
    # 1) Collect docs
    docs = aggregator.collect_comprehensive_data([query], sources=sources)
    texts, meta = aggregator.prepare_for_indexing(docs)

    # 2) Build results preview
    collected_text = ""
    results_cards = []
    for m, t in zip(meta, texts):
        preview = t[:350] + "..." if len(t) > 350 else t
        card = f"""
        <div style="padding:12px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:14px; background:white;">
            <h3 style="margin:0; font-size:16px; font-weight:600; color:#1a73e8;">{m['title']}</h3>
            <a href="{m['url']}" target="_blank" style="font-size:13px; color:#006621; text-decoration:none;">{m['url']}</a>
            <p style="font-size:14px; color:#333; margin-top:8px; line-height:1.5;">{preview}</p>
        </div>
        """
        results_cards.append(card)
        collected_text += f"{t}\n\n"

    # 3) Generate AI-optimized summary
    ai_answer = "⚠️ Could not generate AI answer (no text collected)."
    if collected_text:
        try:
            truncated = collected_text[:3000]  # BART input limit
            summary = summarizer(truncated, max_length=200, min_length=50, do_sample=False)
            ai_answer = summary[0]["summary_text"]
        except Exception as e:
            ai_answer = f"⚠️ Error generating AI answer: {e}"

    # Final combined output
    html_output = f"""
    <div style="padding:16px; border-radius:12px; background:#f9fafb; margin-bottom:20px; box-shadow:0 2px 6px rgba(0,0,0,0.05);">
        <h2 style="margin:0; font-size:18px; color:#111827;">🤖 AI Answer</h2>
        <p style="font-size:15px; color:#333; margin-top:8px; line-height:1.6;">{ai_answer}</p>
    </div>
    <h2 style="margin-top:0; font-size:18px; color:#111827;">📚 Sources</h2>
    {''.join(results_cards)}
    """
    return html_output


with gr.Blocks(css="""
    .gr-button {font-size:16px !important; border-radius:12px !important; padding:10px 18px !important;}
    .gr-textbox textarea {font-size:16px !important;}
""") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#1a73e8;'>🔍 Smart Search</h1>")
    gr.Markdown("<p style='text-align:center; font-size:16px;'>Enter your query below to get an <b>AI-generated answer</b> followed by structured sources.</p>")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Search Query", placeholder="e.g. climate change impact", lines=2)
        with gr.Column(scale=1):
            sources_input = gr.CheckboxGroup(["web", "wikipedia"], value=["web", "wikipedia"], label="Sources")
            search_btn = gr.Button("🚀 Search", variant="primary")

    results_output = gr.HTML(label="Results")

    # Button interaction with loading state
    def _disable_btn():
        return gr.update(interactive=False, value="⏳ Searching...")

    def _enable_btn():
        return gr.update(interactive=True, value="🚀 Search")

    search_btn.click(
        fn=_disable_btn, 
        inputs=None, 
        outputs=search_btn
    ).then(
        fn=search_and_answer,
        inputs=[query_input, sources_input],
        outputs=[results_output],
        show_progress=True
    ).then(
        fn=_enable_btn,
        inputs=None,
        outputs=search_btn
    )

if __name__ == "__main__":
    demo.launch(share=True)


