#!/usr/bin/env python3
"""Lightweight tool execution server for Qwen Local chat UI.

Run alongside the chat server:
    python3 tools_server.py

Provides file read/write and directory listing inside ~/qwen-workspace/
(paths are sandbox-checked). Also exposes run_shell, which executes
arbitrary commands with your privileges — it is NOT sandboxed.
Only run this server while using the chat UI; stop it when not in use.
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
        # Requests from the chat UI carry an Origin header; direct/curl calls don't.
        # Reject anything that claims a different origin.
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
