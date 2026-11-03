import json
import unittest
from unittest.mock import patch
from web_data import validate_url, PAGE_HOSTS, chunks, extract_html, cat_fact, public_get
from rag import RAGAgent


class RAGTests(unittest.TestCase):
    def test_url_scope(self):
        validate_url("https://www.mathsisfun.com/algebra/linear-equations.html", PAGE_HOSTS)
        for url in ["file:///etc/passwd", "http://www.mathsisfun.com", "https://localhost/",
                    "https://www.mathsisfun.com.attacker.test/", "https://user:secret@www.mathsisfun.com/"]:
            with self.assertRaises(ValueError):
                validate_url(url, PAGE_HOSTS)

    def test_chunk_coverage(self):
        text = "".join(str(i % 10) for i in range(2501))
        parts = chunks(text, 900, 120)
        restored = parts[0] + "".join(p[120:] for p in parts[1:])
        self.assertEqual(restored, text)
        self.assertTrue(all(len(p) <= 900 for p in parts))
        with self.assertRaises(ValueError): chunks(text, 100, 100)

    def test_html_cleaning(self):
        html = '<html><title>Lesson</title><body><nav>Navigation</nav><main><h1>Equations</h1><p>' + 'Study algebra. ' * 20 + '</p><script>bad()</script></main></body></html>'
        title, text = extract_html(html)
        self.assertEqual(title, "Lesson")
        self.assertIn("Equations", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn("Navigation", text)

    def test_cat_api_validation(self):
        for body, ctype in [(b'not JSON', 'application/json'), (b'{}', 'application/json'),
                            (b'{"fact": ""}', 'application/json'), (b'<html>', 'text/html')]:
            with patch("web_data.public_get", return_value=(body, ctype, "https://catfact.ninja/fact")):
                with self.assertRaises(ValueError): cat_fact()
        with patch("web_data.public_get", return_value=(b'{"fact":"Example fact"}', 'application/json', 'https://catfact.ninja/fact')):
            self.assertEqual(cat_fact()["method"], "GET")

    def test_empty_source_does_not_invent_matches(self):
        rag = RAGAgent.__new__(RAGAgent)
        rag.sources = {}
        self.assertEqual(rag.search("question")["matches"], [])

    def test_timeout_is_safe(self):
        import requests
        with patch("web_data.requests.get", side_effect=requests.Timeout()):
            with self.assertRaises(RuntimeError):
                public_get("https://catfact.ninja/fact", {"catfact.ninja"})

    def test_latest_version_filter(self):
        from unittest.mock import Mock
        rag = RAGAgent.__new__(RAGAgent)
        rag.sources = {"https://www.mathsisfun.com/example": "current-version"}
        rag.namespace, rag.host = "test", "https://test.pinecone.io"
        rag.pc = Mock()
        rag.pc.embed.return_value = ([[0.1] * 1024], {})
        rag.pc.request.return_value = {"matches": []}
        rag.search("question")
        payload = rag.pc.request.call_args.args[2]
        self.assertEqual(payload["filter"], {"source_version": {"$in": ["current-version"]}})


if __name__ == "__main__":
    unittest.main()
