"""管理类写接口的可选令牌鉴权与 CORS 来源配置；通过真实 ASGI 路由验证，不加载模型。"""
import asyncio
import unittest
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from app import server

PROTECTED = [
    ("POST", "/v1/voiceprint/register"),
    ("DELETE", "/v1/voiceprint?id=a"),
    ("POST", "/v1/voiceprint/reload"),
    ("POST", "/v1/meeting/recordings/delete"),
    ("PUT", "/v1/kws/keywords"),
]


async def call(method, path, headers=(), body=b""):
    path, _, query = path.partition("?")
    messages = []
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "scheme": "http",
             "method": method, "path": path, "raw_path": path.encode(), "query_string": query.encode(),
             "root_path": "", "server": ("test", 80), "client": ("127.0.0.1", 1),
             "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                         *[(k.lower().encode(), v.encode()) for k, v in headers]]}
    await server.app(scope, receive, send)
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


class AdminTokenTest(unittest.TestCase):
    def setUp(self):
        # 写接口在鉴权通过后都落到无副作用的替身上：删除返回 404，重载只计数。
        self.patches = [
            patch.object(server, "delete_voiceprint_by_id", return_value=None),
            patch.object(server.service, "reload_voiceprints"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def status(self, method, path, token=None, authorization=None):
        headers = [("Authorization", authorization)] if authorization is not None else []
        with patch.object(server, "ADMIN_TOKEN", token or ""):
            return asyncio.run(call(method, path, headers))

    def test_without_admin_token_write_endpoints_stay_open(self):
        self.assertEqual(self.status("DELETE", "/v1/voiceprint?id=a"), 404)
        self.assertEqual(self.status("POST", "/v1/voiceprint/reload"), 200)
        for method, path in PROTECTED:
            with self.subTest(path=path):
                self.assertNotEqual(self.status(method, path), 401)

    def test_missing_or_wrong_token_is_rejected(self):
        for method, path in PROTECTED:
            for authorization in (None, "", "Bearer", "Bearer wrong", "Basic secret", "secret", "Bearer secret2"):
                with self.subTest(path=path, authorization=authorization):
                    self.assertEqual(self.status(method, path, "secret", authorization), 401)
        server.service.reload_voiceprints.assert_not_called()
        server.delete_voiceprint_by_id.assert_not_called()

    def test_correct_token_passes(self):
        self.assertEqual(self.status("DELETE", "/v1/voiceprint?id=a", "secret", "Bearer secret"), 404)
        self.assertEqual(self.status("POST", "/v1/voiceprint/reload", "secret", "bearer  secret"), 200)
        server.delete_voiceprint_by_id.assert_called_once_with("a")
        for method, path in PROTECTED:
            with self.subTest(path=path):
                self.assertNotEqual(self.status(method, path, "secret", "Bearer secret"), 401)

    def test_read_endpoints_do_not_require_token(self):
        with patch.object(server, "load_voiceprint_index", return_value={}):
            self.assertEqual(self.status("GET", "/v1/voiceprint/list", "secret"), 200)
            self.assertEqual(self.status("GET", "/v1/voiceprint/exists?id=a", "secret"), 200)
        self.assertEqual(self.status("GET", "/v1/live/metrics", "secret"), 200)


class CorsOriginsTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(server._parse_cors_allow_origins(None), ["*"])
        self.assertEqual(server._parse_cors_allow_origins(" , "), ["*"])
        self.assertEqual(server._parse_cors_allow_origins("https://a.example, http://b.example:8080/ ,"),
                         ["https://a.example", "http://b.example:8080"])

    def test_middleware_uses_configured_origins(self):
        cors = next(m for m in server.app.user_middleware if m.cls is CORSMiddleware)
        self.assertIs(cors.kwargs["allow_origins"], server.CORS_ALLOW_ORIGINS)


if __name__ == "__main__":
    unittest.main()
