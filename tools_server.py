#!/usr/bin/env python3
"""Lightweight tool execution server for Qwen Local chat UI.

Run alongside the chat server:
    python3 tools_server.py

Tools:
  read_file / write_file / list_directory  — workspace-sandboxed (~/qwen-workspace/)
  edit_file     — surgical find-replace on ANY absolute path (same trust as run_shell)
  run_shell     — arbitrary shell commands; NOT sandboxed
  screenshot    — headless Chromium screenshot of a localhost URL → workspace PNG
  eval_js       — run JS in a headless page and return the result
  get_page_html — return rendered DOM of a localhost URL

Only run this server while actively using the chat UI.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, subprocess
from pathlib import Path

WORKSPACE = Path.home() / 'qwen-workspace'
WORKSPACE.mkdir(exist_ok=True)
PORT = 8081
ALLOWED_ORIGIN = 'http://localhost:8080'

# Commands that are never allowed regardless of context
BLOCKED_PATTERNS = ['rm -rf /', 'rm -rf ~', 'sudo rm', '> /dev/sd', 'mkfs', ':(){:|:&};:']


class ToolHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f'[tool] {fmt % args}')

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _check_origin(self):
        """Return True only if the request comes from the expected origin."""
        origin = self.headers.get('Origin', '')
        host   = self.headers.get('Host', '')
        # Requests from the chat UI always carry Origin (browsers attach it to
        # every cross-origin POST, including text/plain "simple" requests that
        # skip preflight). Requests with *no* Origin (curl, local scripts) are
        # intentionally allowed — don't "fix" this: it would break CLI access.
        if origin and origin != ALLOWED_ORIGIN:
            return False
        # Reject requests routed to an unexpected Host (DNS-rebinding defence).
        if host and not host.startswith(('localhost:', '127.0.0.1:')):
            return False
        return True

    def do_POST(self):
        if not self._check_origin():
            self.send_response(403)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error":"Forbidden"}')
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError:
            body = {}
        tool = self.path.strip('/')
        result = self._dispatch(tool, body)
        data = json.dumps(result).encode()
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _safe_path(self, rel):
        """Resolve path and verify it's strictly inside WORKSPACE."""
        ws = WORKSPACE.resolve()
        p  = (WORKSPACE / (rel or '')).resolve()
        # Use relative_to() — startswith() is wrong because /a/b-extra starts with /a/b
        try:
            p.relative_to(ws)
        except ValueError:
            raise ValueError(f'Path escapes workspace: {rel!r}')
        return p

    def _dispatch(self, tool, args):
        try:
            if tool == 'read_file':
                p = self._safe_path(args.get('path', ''))
                if not p.exists():
                    return {'error': f'File not found: {p.name}'}
                if p.is_dir():
                    return {'error': f'{p.name} is a directory'}
                return {'content': p.read_text(errors='replace'), 'path': str(p), 'bytes': p.stat().st_size}

            elif tool == 'write_file':
                p = self._safe_path(args.get('path', 'output.txt'))
                p.parent.mkdir(parents=True, exist_ok=True)
                content = args.get('content', '')
                p.write_text(content)
                return {'success': True, 'path': str(p), 'bytes': len(content.encode())}

            elif tool == 'list_directory':
                p = self._safe_path(args.get('path', ''))
                if not p.exists():
                    return {'error': f'Directory not found: {p.name or "/"}'}
                entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
                return {
                    'path': str(p),
                    'entries': [
                        {'name': e.name, 'type': 'dir' if e.is_dir() else 'file',
                         'size': e.stat().st_size if e.is_file() else None}
                        for e in entries
                    ]
                }

            elif tool == 'edit_file':
                # Surgical find-replace — accepts absolute paths (same trust as run_shell)
                path = args.get('path', '')
                old_text = args.get('old_text', '')
                new_text = args.get('new_text', '')
                if not path:
                    return {'error': 'path is required'}
                p = Path(path).expanduser()
                if not p.exists():
                    return {'error': f'File not found: {path}'}
                content = p.read_text(errors='replace')
                count = content.count(old_text)
                if count == 0:
                    return {'error': 'old_text not found in file — check for exact whitespace/indentation match'}
                if count > 1:
                    return {'error': f'old_text matches {count} locations — make it more specific'}
                p.write_text(content.replace(old_text, new_text, 1))
                return {'success': True, 'path': str(p)}

            elif tool == 'run_shell':
                cmd = args.get('command', '').strip()
                if not cmd:
                    return {'error': 'Empty command'}
                if any(b in cmd for b in BLOCKED_PATTERNS):
                    return {'error': f'Command blocked for safety'}
                r = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    cwd=WORKSPACE, timeout=30
                )
                return {'stdout': r.stdout, 'stderr': r.stderr, 'returncode': r.returncode}

            elif tool in ('screenshot', 'eval_js', 'get_page_html'):
                url = args.get('url', 'http://localhost:8080/chat.html')
                if not url.startswith(('http://localhost:', 'http://127.0.0.1:')):
                    return {'error': 'Browser tools are restricted to localhost URLs'}
                from playwright.sync_api import sync_playwright
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True)
                    page = browser.new_page(viewport={'width': 1280, 'height': 800})
                    page.goto(url, wait_until='networkidle', timeout=15000)

                    if tool == 'screenshot':
                        filename = args.get('filename', 'screenshot.png')
                        out = WORKSPACE / filename
                        page.screenshot(path=str(out), full_page=args.get('full_page', False))
                        browser.close()
                        return {'path': str(out), 'filename': filename, 'bytes': out.stat().st_size}

                    elif tool == 'eval_js':
                        script = args.get('script', 'document.title')
                        result = page.evaluate(script)
                        browser.close()
                        return {'result': result}

                    else:  # get_page_html
                        html = page.content()
                        browser.close()
                        cap = 40000
                        return {'html': html[:cap], 'total_length': len(html),
                                'truncated': len(html) > cap}

            else:
                return {'error': f'Unknown tool: {tool!r}'}

        except ValueError as e:
            return {'error': str(e)}
        except subprocess.TimeoutExpired:
            return {'error': 'Command timed out after 30 seconds'}
        except PermissionError as e:
            return {'error': f'Permission denied: {e}'}
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    print(f'Tool server running on http://127.0.0.1:{PORT}')
    print(f'Workspace: {WORKSPACE}')
    print('Ctrl+C to stop\n')
    try:
        HTTPServer(('127.0.0.1', PORT), ToolHandler).serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
