"""Collect article URLs from a Common Crawl CC-NEWS WARC archive."""

import argparse
import gzip
import io
import json
import ssl
import urllib.request
from typing import Generator

try:
    from warcio.archiveiterator import ArchiveIterator
except ImportError as exc:
    ArchiveIterator = None
    WARCIO_IMPORT_ERROR = exc
else:
    WARCIO_IMPORT_ERROR = None


COMMON_CRAWL_BASE = "https://data.commoncrawl.org/"
DEFAULT_WARC_PATHS_URL = (
    "https://data.commoncrawl.org/crawl-data/CC-NEWS/2025/12/warc.paths.gz"
)

# If HTTPS certificate verification fails locally, run:
# python3 back-end/testingcode/crawlparse.py --limit 10 --insecure


def fetch_bytes(url: str, insecure: bool = False) -> bytes:
    with urllib.request.urlopen(
        url, context=build_ssl_context(insecure=insecure)
    ) as response:
        return response.read()


def iter_warc_paths(
    warc_paths_url: str,
    insecure: bool = False,
) -> Generator[str, None, None]:
    compressed = fetch_bytes(warc_paths_url, insecure=insecure)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz_file:
        for raw_line in gz_file:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield line


def build_warc_url(warc_path: str) -> str:
    # This builds the URL of the archive file itself, not the article URLs inside it.
    return urllib.request.urljoin(COMMON_CRAWL_BASE, warc_path)


def iter_warc_target_uris(
    warc_url: str,
    insecure: bool = False,
) -> Generator[str, None, None]:
    if ArchiveIterator is None:
        raise RuntimeError(
            "warcio is required to parse WARC files. "
            "Install it with: pip install -r back-end/testingcode/testing-requirements.txt"
        ) from WARCIO_IMPORT_ERROR

    with urllib.request.urlopen(
        warc_url, context=build_ssl_context(insecure=insecure)
    ) as response:
        # `warcio` handles the WARC record format for us, which is safer and easier
        # to maintain than manually parsing headers and record boundaries.
        for record in ArchiveIterator(response):
            if record.rec_type != "response":
                continue

            target_uri = record.rec_headers.get_header("WARC-Target-URI")
            if target_uri:
                yield target_uri


def build_ssl_context(insecure: bool = False) -> ssl.SSLContext:
    # This affects HTTPS verification while downloading files.
    # It does not change the WARC parsing logic itself.
    #
    # Normal behavior: verify the server certificate.
    # Insecure behavior: skip verification, which can help if the local Python trust
    # store is misconfigured and HTTPS requests fail even for valid URLs.
    if insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def extract_urls(
    warc_paths_url: str,
    warc_index: int,
    limit: int,
    insecure: bool,
) -> dict:
    # First parse the index file to discover which WARC archives exist.
    warc_paths = list(iter_warc_paths(warc_paths_url, insecure=insecure))
    if not warc_paths:
        raise RuntimeError("No WARC paths found in the Common Crawl index.")
    if warc_index < 0 or warc_index >= len(warc_paths):
        raise IndexError(
            f"warc_index {warc_index} is out of range for {len(warc_paths)} WARC paths."
        )

    warc_path = warc_paths[warc_index]
    warc_url = build_warc_url(warc_path)

    # Important distinction:
    # - `warc_url` is the URL of the archive file on Common Crawl
    # - `target_uri` values are the original article/page URLs stored inside that archive
    urls = []
    seen = set()
    for target_uri in iter_warc_target_uris(warc_url, insecure=insecure):
        if target_uri in seen:
            continue
        seen.add(target_uri)
        urls.append(target_uri)
        if len(urls) >= limit:
            break

    return {
        "warc_paths_url": warc_paths_url,
        "warc_path": warc_path,
        "warc_url": warc_url,
        "url_count": len(urls),
        "urls": urls,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract article URLs from a Common Crawl CC-NEWS WARC file."
    )
    parser.add_argument(
        "--warc-paths-url",
        default=DEFAULT_WARC_PATHS_URL,
        help="Gzipped Common Crawl WARC path listing to parse.",
    )
    parser.add_argument(
        "--warc-index",
        type=int,
        default=0,
        help="Zero-based index of the WARC path to inspect from warc.paths.gz.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of unique article URLs to print.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification if your local Python trust store is broken.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # `main()` is the command-line entry point. It reads user options, runs the full
    # extraction pipeline, and prints the result as JSON.
    result = extract_urls(
        warc_paths_url=args.warc_paths_url,
        warc_index=args.warc_index,
        limit=args.limit,
        insecure=args.insecure,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
