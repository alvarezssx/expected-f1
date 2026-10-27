"""Yerel web sunucusu: python server.py  ->  http://localhost:8765

Yalnızca bu bilgisayardan erişilir (127.0.0.1). Ek korumalar:
  - Host başlığı denetimi (DNS rebinding'e karşı)
  - POST yalnızca JSON + aynı köken (başka sitelerin tarayıcı üzerinden istek atmasına karşı)
  - istek gövdesi boyut sınırı, girdi doğrulama
  - hata ayrıntıları yalnızca sunucu konsoluna yazılır, yanıta konmaz
"""
from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from f1tahmin.engine import Engine
from f1tahmin.odds import MARKET_K

STATIC = Path(__file__).resolve().parent / "static"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ALLOWED_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
MAX_BODY = 64 * 1024

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
}


class BadRequest(Exception):
    pass


print("Veri yükleniyor ve model kalibre ediliyor...", flush=True)
ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            self._json({"error": "izin verilmeyen host"}, 403)
            return False
        return True

    def _guard(self, fn):
        try:
            fn()
        except BadRequest as e:
            self._json({"error": str(e)}, 400)
        except Exception:  # ayrıntı yalnızca konsola: yanıtta yerel dosya yolları sızmasın
            traceback.print_exc()
            self._json({"error": "Sunucu hatası (ayrıntı sunucu konsolunda)."}, 500)

    def do_GET(self):
        if not self._host_ok():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}

        def handle():
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/calendar":
                self._json({"races": ENGINE.calendar_view(), "next": ENGINE.next_round(), "temps": ENGINE.temps})
            elif url.path == "/api/predict":
                rnd = _round(q.get("round") or ENGINE.next_round() or max(ENGINE.done))
                stage = q.get("stage", "")
                if stage not in ("", "pre"):
                    raise BadRequest("bilinmeyen tahmin anı")
                self._json(ENGINE.predict(rnd, "simulated" if stage == "pre" else "auto"))
            elif url.path == "/api/backtest":
                self._json(ENGINE.backtest())
            elif url.path == "/api/championship":
                self._json(ENGINE.championship())
            else:
                self._json({"error": "bulunamadı"}, 404)
        self._guard(handle)

    def do_POST(self):
        if not self._host_ok():
            return
        url = urlparse(self.path)

        def handle():
            # Başka sitelerden gelen istekleri reddet: JSON zorunluluğu tarayıcıyı ön kontrole (CORS) zorlar,
            # Origin varsa bizim adresimiz olmalı.
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                raise BadRequest("Content-Type application/json olmalı")
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc not in ALLOWED_HOSTS:
                raise BadRequest("izin verilmeyen köken")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise BadRequest("istek çok büyük")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeDecodeError):
                raise BadRequest("geçersiz JSON")
            if not isinstance(body, dict):
                raise BadRequest("geçersiz istek")

            if url.path == "/api/odds":
                market = body.get("market", "win")
                if market not in MARKET_K:
                    raise BadRequest("bilinmeyen piyasa")
                raw = body.get("odds") or {}
                if not isinstance(raw, dict) or len(raw) > 60:
                    raise BadRequest("geçersiz oran listesi")
                odds = {}
                for d, v in raw.items():
                    if v in (None, ""):
                        continue
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        raise BadRequest(f"geçersiz oran: {str(d)[:40]}")
                    if not (1.0 < f < 100000):
                        raise BadRequest(f"oran aralık dışında: {str(d)[:40]}")
                    odds[str(d)[:60]] = f
                self._json(ENGINE.analyze_odds(_round(body.get("round")), market, odds))
            elif url.path == "/api/refresh":
                ENGINE.load(refresh=True)
                self._json({"ok": True, "next": ENGINE.next_round()})
            else:
                self._json({"error": "bulunamadı"}, 404)
        self._guard(handle)

    def do_HEAD(self):
        if self._host_ok():
            self._send(200, b"", "text/html; charset=utf-8")

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            sys.stderr.write("%s\n" % (fmt % args))


def _round(v) -> int:
    try:
        rnd = int(v)
    except (TypeError, ValueError):
        raise BadRequest("geçersiz yarış numarası")
    if rnd not in {int(r["round"]) for r in ENGINE.calendar}:
        raise BadRequest("takvimde olmayan yarış")
    return rnd


if __name__ == "__main__":
    print(f"Hazır: http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
