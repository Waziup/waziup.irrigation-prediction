
import mimetypes
import errno
import os
from pathlib import Path
import socket
import stat
import sys
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
from urllib.parse import unquote, urlparse
import traceback

from dotenv import load_dotenv


sockAddr = ""
_server = None


def _positive_int_setting(name, default):
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


# API requests are small form/JSON payloads. Operators can tune these without
# allowing invalid configuration to disable the safety bounds.
MAX_REQUEST_BODY_BYTES = _positive_int_setting('MAX_REQUEST_BODY_BYTES', 1024 * 1024)
REQUEST_READ_TIMEOUT_SECONDS = _positive_int_setting('REQUEST_READ_TIMEOUT_SECONDS', 20)

# ----------------- #

routing = {}
routing["GET"] = {}
routing["POST"] = {}
routing["PUT"] = {}
routing["DELETE"] = {}


def resolve_static_path(request_path, static_root=None):
    """Resolve a /dist URL below the static root, rejecting traversal/symlinks."""
    root = (Path(static_root) if static_root else Path.cwd() / "dist").resolve()
    decoded_path = unquote(urlparse(request_path).path)
    if not decoded_path.startswith("/dist/"):
        return None
    relative_path = decoded_path[len("/dist/"):]
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate

# -------------------#


def routerGET(path, func):
    global routing
    routing["GET"][path] = func

# ---------#


def routerPOST(path, func):
    global routing
    routing["POST"][path] = func

# ---------#


def routerPUT(path, func):
    global routing
    routing["PUT"][path] = func

# ---------#


def routerDELETE(path, func):
    global routing
    routing["DELETE"][path] = func


# ---------#

# -------------------#

class HTTPHandler(BaseHTTPRequestHandler):
    # protocol_version = "HTTP/1.1"
    # ---------------#
    def setup(self):
        super().setup()
        settimeout = getattr(self.connection, 'settimeout', None)
        if settimeout is not None:
            settimeout(REQUEST_READ_TIMEOUT_SECONDS)

    def address_string(self):
        # Unix-domain clients do not have the (host, port) tuple expected by
        # BaseHTTPRequestHandler's default timeout/error logger.
        return 'local-unix-client'

    def callAPI(self, method="GET", body=""):
        inPath = urlparse(self.path).path
        routPath = ""

        try:
            # Try matching the incoming path with the registered routes
            for key in routing.get(method, {}):
                if re.match(r"^" + key + "$", inPath):
                    routPath = key
                    break

            if not routPath:
                allowed = sorted(verb for verb, routes in routing.items()
                                 if any(re.match(r"^" + key + "$", inPath)
                                        for key in routes))
                if allowed:
                    self.send(405, b"Method not allowed", ["text/plain"],
                              extra_headers={"Allow": ", ".join(allowed)})
                else:
                    self.send(404, b"Not found", ["text/plain"])
                return

            # Call the registered route function
            resCode, resBody, resHeaders = routing[method][routPath](
                self.path, body)

        except Exception as e:
            print(f"Exception in callAPI(): {e}")
            traceback.print_exc()
            resCode = 500
            resBody = b"Internal server error"
            resHeaders = ["text/plain"]

        # Send the response regardless
        self.send(resCode, resBody, resHeaders)

    # ---------------#

    def serve_static_file(self):
        file_path = resolve_static_path(self.path)
        if file_path is not None and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or 'application/octet-stream'
            with file_path.open("rb") as file:
                self.send(200, file.read(), [mime_type])
        else:
            self.send(404, b"File not found", ["text/plain"])

    # ----------------------#

    def do_GET(self):
        if self.path.startswith("/dist/"):  # Adjust the folder as needed
            self.serve_static_file()
        else:
            self.callAPI()

    # ---------------#

    def has_trusted_origin(self):
        origins = self.headers.get_all('Origin', [])
        if not origins:
            # Preserve trusted gateway/service clients that do not operate in
            # a browser origin model.
            return True
        hosts = self.headers.get_all('Host', [])
        if len(origins) != 1 or len(hosts) != 1:
            return False
        origin = urlparse(origins[0].strip())
        if origin.scheme not in {'http', 'https'} or not origin.netloc:
            return False
        if origin.username is not None or origin.password is not None:
            return False
        if origin.path not in {'', '/'} or origin.params or origin.query or origin.fragment:
            return False
        if origin.netloc.lower() == hosts[0].strip().lower():
            return True

        # Development reverse proxies commonly replace Host while preserving
        # the browser's Origin. Permit only explicitly configured full origins;
        # production remains same-origin when this setting is absent.
        normalized_origin = f'{origin.scheme.lower()}://{origin.netloc.lower()}'
        configured = {
            value.strip().lower().rstrip('/')
            for value in os.getenv('TRUSTED_BROWSER_ORIGINS', '').split(',')
            if value.strip()
        }
        return normalized_origin in configured

    def dispatch_body(self, method):
        if not self.has_trusted_origin():
            if urlparse(self.path).path.startswith('/api/'):
                self.send(403, b'{"status":"error","error":"Cross-origin request rejected"}',
                          ['application/json'])
            else:
                self.send(403, b'Cross-origin request rejected', ['text/plain'])
            return
        lengths = self.headers.get_all('Content-Length', [])
        transfer_encodings = self.headers.get_all('Transfer-Encoding', [])
        if transfer_encodings:
            # This server does not implement chunk decoding. Never interpret a
            # chunked request as an empty fixed-length body. Conflicting framing
            # is malformed and may indicate request smuggling.
            if lengths:
                self.send(400, b'Conflicting request framing', ['text/plain'])
            else:
                self.send(501, b'Transfer-Encoding is not supported', ['text/plain'])
            return
        # Missing length retains the existing empty-body API behavior. Reject
        # ambiguous or malformed lengths before reading or invoking a handler.
        value = lengths[0].strip() if lengths else '0'
        if len(lengths) > 1 or re.fullmatch(r'[0-9]+', value) is None:
            self.send(400, b'Invalid Content-Length', ['text/plain'])
            return
        try:
            size = int(value)
            if size > sys.maxsize:
                self.send(400, b'Invalid Content-Length', ['text/plain'])
                return
            if size > MAX_REQUEST_BODY_BYTES:
                self.send(413, b'Request body too large', ['text/plain'])
                return
            body = self.rfile.read(size)
        except socket.timeout:
            self.close_connection = True
            try:
                self.send(408, b'Request body timed out', ['text/plain'])
            except OSError:
                pass
            return
        except (ValueError, OverflowError):
            self.send(400, b'Invalid Content-Length', ['text/plain'])
            return
        if len(body) != size:
            self.send(400, b'Incomplete request body', ['text/plain'])
            return
        self.callAPI(method, body)

    def do_POST(self):
        self.dispatch_body('POST')

    # ---------------#

    def do_PUT(self):
        self.dispatch_body('PUT')

    # ---------------#

    def do_DELETE(self):
        self.dispatch_body('DELETE')

    # ---------------#

    def send(self, code, reply, resHeaders, extra_headers=None):
        self.client_address = (
            '', )  # avoid exception in BaseHTTPServer.py log_message()
        self.send_response(code)

        # We may need more fixes here, this kind of header is just for content-type
        if len(resHeaders) > 0:
            for h in resHeaders:
                self.send_header('Content-type', h)
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)

        # Do not cache HTML so dashboard updates become visible immediately.
        # Prevent its proxy and the browser from retaining an older UI or API
        # response after source/configuration changes.
        self.send_header('Cache-Control', 'no-store, max-age=0')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')

        # self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(reply)


# ----------------------#


def start():
    global sockAddr, _server
    load_dotenv()
    sockAddr = os.getenv("Proxy_URL")

    if not sockAddr:
        raise ValueError('Proxy_URL must specify a Unix socket path')
    path = Path(sockAddr)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISSOCK(existing.st_mode):
            raise FileExistsError(f'Refusing to replace non-socket path: {path}')
        # Only reclaim a stale socket, never disconnect another running server.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(1)
            try:
                probe.connect(str(path))
            except OSError as error:
                if error.errno != errno.ECONNREFUSED:
                    raise
            else:
                raise FileExistsError(f'Unix socket already in use: {path}')
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (existing.st_dev, existing.st_ino):
            raise FileExistsError(f'Unix socket changed during startup: {path}')
        path.unlink()

    class UnixHTTPServer(ThreadingHTTPServer):
        address_family = socket.AF_UNIX

    # Construct the correct socket directly, instead of leaking an unused TCP
    # socket when replacing ThreadingHTTPServer.socket.
    server = UnixHTTPServer(str(path), HTTPHandler, bind_and_activate=False)
    owned = None
    try:
        server.socket.settimeout(20)
        server.socket.bind(str(path))
        bound = path.lstat()
        owned = (bound.st_dev, bound.st_ino)
        # The host WaziGate process may run as a different user than this container.
        # Ensure world read/write on the unix socket so app proxy requests do not fail with EACCES.
        os.chmod(path, 0o666)
        server.server_activate()
        _server = server
        server.serve_forever()
    finally:
        if _server is server:
            _server = None
        server.server_close()
        # Do not remove a replacement created by another process during shutdown.
        if owned is not None:
            try:
                current = path.lstat()
                if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == owned:
                    path.unlink()
            except FileNotFoundError:
                pass

# Just a wrapper to start the server with recovery
# This function will restart the server if it crashes


def start_with_recovery():
    while True:
        try:
            print("Attempting to start HTTP server...")
            start()  # real server function
        except Exception as e:
            print("Server crashed with error:")
            traceback.print_exc()
            print("Restarting in 5 seconds...")
            time.sleep(5)
        else:
            print("Server exited normally — breaking out.")
            break  # Exit if server stops on purpose


def stop():
    """Stop the active Unix-socket HTTP server, if one is running."""
    server = _server
    if server is not None:
        server.shutdown()
