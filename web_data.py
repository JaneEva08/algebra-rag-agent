"""Bounded, allowlisted public HTTP reads; no cookies or API credentials."""
import json
import re
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

PAGE_HOSTS = {"www.mathsisfun.com", "mathsisfun.com"}


def validate_url(url, hosts):
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in hosts or p.username or p.password or p.port not in (None, 443):
        raise ValueError("Разрешены только HTTPS-страницы настроенных учебных сайтов.")
    return url


def public_get(url, hosts):
    for _ in range(4):
        validate_url(url, hosts)
        try:
            with requests.get(url, timeout=25, allow_redirects=False, stream=True,
                              headers={"User-Agent": "AlgebraTutorStudy/1.0"}) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, r.headers.get("Location", ""))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"Открытый источник вернул HTTP {r.status_code}.")
                parts, size = [], 0
                for part in r.iter_content(65536):
                    size += len(part)
                    if size > 2_000_000:
                        raise ValueError("Источник больше 2 МБ.")
                    parts.append(part)
                return b"".join(parts), r.headers.get("Content-Type", ""), url
        except requests.RequestException:
            raise RuntimeError("Открытый источник недоступен или превышено время ожидания.") from None
    raise RuntimeError("Слишком много перенаправлений.")


def extract_html(content):
    soup = BeautifulSoup(content, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Учебная страница"
    body = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body or soup
    for tag in body.find_all(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", body.get_text(" ", strip=True))
    if len(text) < 100:
        raise ValueError("На странице недостаточно учебного текста.")
    return title, text


def chunks(text, size=900, overlap=120):
    if not 0 <= overlap < size:
        raise ValueError("Некорректный размер перекрытия.")
    result = []
    for start in range(0, len(text), size - overlap):
        result.append(text[start:start + size])
        if start + size >= len(text):
            break
    return result


def fetch_page(url):
    body, content_type, final_url = public_get(url, PAGE_HOSTS)
    if "text/html" not in content_type.lower():
        raise ValueError("Ожидалась HTML-страница.")
    title, text = extract_html(body)
    return final_url, title, text


def cat_fact():
    body, content_type, final_url = public_get("https://catfact.ninja/fact", {"catfact.ninja"})
    if "json" not in content_type.lower():
        raise ValueError("API вернул не JSON.")
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeError):
        raise ValueError("API вернул некорректный JSON.") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("fact"), str) or not obj["fact"].strip():
        raise ValueError("API не вернул непустой факт.")
    return {"fact": obj["fact"], "source": final_url, "method": "GET", "status": 200,
            "verification": "unverified_external_claim"}
