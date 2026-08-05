from __future__ import annotations

import http.server
import json
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    root: Path

    def do_GET(self) -> None:
        if self.path == "/api/plugins":
            data = (self.root / "plugins" / "registry.json").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path in ("/", "/index.html"):
            data = (self.root / "ui" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


def serve(root: Path, port: int = 8765) -> None:
    Handler.root = root
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"UI available at http://127.0.0.1:{port} (Ctrl+C to stop)")
    server.serve_forever()
