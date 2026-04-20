#!/usr/bin/env python3
# Auto-generated sidecar server for EV_results.html.
# Run: python3 .bin/EV_results_serve.py   (or double-click EV_results.sh)
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import os, json, http.server, socketserver

PAGE   = '.bin/EV_results.html'
PORT   = 8080
FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # results root, not .bin

_parquet_to_json: Callable[[str], list[dict[str, Any]]] | None = None

try:
    import polars as pl
    def _polars_loader(path: str) -> list[dict[str, Any]]:
        return pl.read_parquet(path).to_dicts()
    _parquet_to_json = _polars_loader
except ImportError:
    try:
        import pyarrow.parquet as pq
        def _pyarrow_loader(path: str) -> list[dict[str, Any]]:
            return pq.read_table(path).to_pydict()
        _parquet_to_json = _pyarrow_loader
    except ImportError:
        pass

class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=FOLDER, **kw)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path.endswith('.parquet') and _parquet_to_json:
            fpath = os.path.join(FOLDER, path.lstrip('/').replace('/', os.sep))
            if os.path.isfile(fpath):
                try:
                    data = _parquet_to_json(fpath)
                    body = json.dumps(data).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(body)
                    return
                except Exception as e:
                    print(f'[serve] parquet error {fpath}: {e}')
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

if __name__ == '__main__':
    import os; os.chdir(FOLDER)
    print(f'Serving  {FOLDER}')
    print(f'Open     http://localhost:{PORT}/{PAGE}')
    print('Stop     Ctrl+C')
    try:
        with socketserver.TCPServer(('', PORT), _Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    except OSError as exc:
        if getattr(exc, 'errno', None) in (98, 10048) or 'address' in str(exc).lower():
            print(f'Port {PORT} busy -- a server is already running.')
            print(f'Open http://localhost:{PORT}/{PAGE} in your browser.')
        else:
            raise
