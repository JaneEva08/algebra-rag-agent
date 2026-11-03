import json, math
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
MODEL="multilingual-e5-large"
DIMENSION=1024
API_VERSION="2025-10"

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class APIError(RuntimeError):
    def __init__(self,status,message):
        self.status=status
        super().__init__(message)

class Pinecone:
    def __init__(self, key):
        self.key = key

    def request(self, host, route, body=None):
        # Never send the API key outside Pinecone, including via redirects.
        parsed = urlparse(host)
        if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".pinecone.io"):
            raise ValueError("Недопустимый хост Pinecone.")
        req = Request(host + route, data=None if body is None else json.dumps(body).encode(),
                      headers={"Api-Key": self.key, "X-Pinecone-Api-Version": API_VERSION,
                               "Content-Type": "application/json"})
        try:
            with build_opener(NoRedirect()).open(req, timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            # Do not expose request headers or arbitrary service response bodies.
            raise APIError(exc.code, f"Pinecone HTTP {exc.code} ({route}).") from None
        except (URLError, OSError):
            raise APIError(0, "Pinecone недоступен или превышено время ожидания.") from None

    def embed(self, texts, input_type):
        response = self.request("https://api.pinecone.io", "/embed", {
            "model": MODEL, "parameters": {"input_type": input_type, "truncate": "END"},
            "inputs": [{"text": text} for text in texts]})
        vectors = [item["values"] for item in response["data"]]
        validate_vectors(vectors, len(texts))
        return vectors, response.get("usage", {})


def validate_vectors(vectors, count):
    if len(vectors) != count or any(len(v) != DIMENSION for v in vectors):
        raise ValueError("Количество или размерность эмбеддингов не совпадает с ожидаемой.")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for v in vectors for x in v):
        raise ValueError("Эмбеддинг содержит некорректные числа.")

