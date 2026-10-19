"""Local web GUI: serves the page and pushes engine frames via Server-Sent Events."""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

WEB_DIR = Path(__file__).parent / "web"


class GuiServer:
    def __init__(self, host: str, port: int, config_message: dict,
                 on_command: Callable[[dict], None] | None = None):
        self.config_json = json.dumps(config_message)
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (WEB_DIR / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    gui._stream(self)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/control" or on_command is None:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    cmd = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self.send_error(400)
                    return
                on_command(cmd)
                self.send_response(204)
                self.end_headers()

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"

    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()

    def publish(self, message: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(message, separators=(",", ":"))
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def _stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        q: queue.Queue = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.add(q)
        try:
            handler.wfile.write(f"data: {self.config_json}\n\n".encode())
            handler.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=2)
                    handler.wfile.write(f"data: {data}\n\n".encode())
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.clients.discard(q)
