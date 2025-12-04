#!/usr/bin/env python3
"""
Fetch Apple Metal documentation using Playwright (headless Chrome).
Saves pages as markdown for offline reference.

Usage:
    pip install playwright markdownify
    playwright install chromium
    python fetch_docs.py
"""

import asyncio
import os
from pathlib import Path

try:
    from playwright.async_api import async_playwright
    from markdownify import markdownify as md
except ImportError:
    print("Installing dependencies...")
    os.system("pip install playwright markdownify")
    os.system("playwright install chromium")
    from playwright.async_api import async_playwright
    from markdownify import markdownify as md

# Documentation URLs to fetch
URLS = [
    ("metal_overview", "https://developer.apple.com/documentation/metal"),
    ("metal_raytracing", "https://developer.apple.com/documentation/metal/metal_sample_code_library/accelerating_ray_tracing_using_metal"),
    ("intersection_functions", "https://developer.apple.com/documentation/metal/mtlintersectionfunctiontable"),
    ("acceleration_structure", "https://developer.apple.com/documentation/metal/mtlaccelerationstructure"),
    ("bounding_box", "https://developer.apple.com/documentation/metal/mtlaccelerationstructureboundingboxgeometrydescriptor"),
    ("raytracing_pipeline", "https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu"),
]

DOCS_DIR = Path(__file__).parent.parent / "docs" / "metal_reference"


async def fetch_page(page, name: str, url: str) -> str:
    """Fetch a single documentation page and convert to markdown."""
    print(f"Fetching {name}: {url}")

    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)

        # Wait for content to load
        await page.wait_for_selector("main", timeout=10000)

        # Get the main content
        content = await page.evaluate("""
            () => {
                const main = document.querySelector('main') || document.body;
                // Remove navigation, headers, footers
                const elementsToRemove = main.querySelectorAll('nav, header, footer, .navigation, .sidebar');
                elementsToRemove.forEach(el => el.remove());
                return main.innerHTML;
            }
        """)

        # Convert to markdown
        markdown = md(content, heading_style="ATX", code_language="swift")

        # Clean up excessive whitespace
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)

        # Add source URL header
        markdown = f"# {name}\n\nSource: {url}\n\n---\n\n{markdown}"

        return markdown

    except Exception as e:
        print(f"  Error fetching {name}: {e}")
        return f"# {name}\n\nError fetching: {e}\n\nURL: {url}"


async def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = await context.new_page()

        all_docs = []

        for name, url in URLS:
            markdown = await fetch_page(page, name, url)

            # Save individual file
            doc_path = DOCS_DIR / f"{name}.md"
            doc_path.write_text(markdown)
            print(f"  Saved to {doc_path}")

            all_docs.append(markdown)

        # Save combined reference
        combined = "\n\n---\n\n".join(all_docs)
        combined_path = DOCS_DIR / "combined_reference.md"
        combined_path.write_text(combined)
        print(f"\nCombined reference saved to {combined_path}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
