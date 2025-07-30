
import mimetypes
import os
from pathlib import Path
import socket
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
from urllib.parse import urlparse
import traceback 

from dotenv import load_dotenv


sockAddr = ""

# ----------------- #

routing = {}
routing["GET"] = {}
routing["POST"] = {}
routing["PUT"] = {}
routing["DELETE"] = {}

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
                raise Exception(f"No route matched for path: {inPath} and method: {method}")

            # Call the registered route function
            resCode, resBody, resHeaders = routing[method][routPath](self.path, body)

        except Exception as e:
            print(f"Exception in callAPI(): {e}")
            traceback.print_exc()
            resCode = 500
            resBody = f"Internal server error:\n{str(e)}".encode("utf-8")
            resHeaders = ["text/plain"]

        # Send the response regardless
        self.send(resCode, resBody, resHeaders)

    # ---------------#

    def serve_static_file(self):
        file_path = Path("." + self.path)  # Convert URL path to file path
        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or 'application/octet-stream'

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()

            with file_path.open("rb") as file:
                self.wfile.write(file.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")

    # ----------------------#

    def do_GET(self):
        if self.path.startswith("/dist/"):  # Adjust the folder as needed
            self.serve_static_file()
        else:
            self.callAPI()

    # ---------------#

    def do_POST(self):
        size = int(self.headers.get('Content-length', 0))
        body = self.rfile.read(size)
        # fields = urlparse.parse_qs(body)

        self.callAPI("POST", body)

    # ---------------#

    def do_PUT(self):
        size = int(self.headers.get('Content-length', 0))
        body = self.rfile.read(size)
        # fields = urlparse.parse_qs(body)

        self.callAPI("PUT", body)

    # ---------------#

    def do_DELETE(self):
        size = int(self.headers.get('Content-length', 0))
        body = self.rfile.read(size)

        self.callAPI("DELETE", body)

    # ---------------#

    def send(self, code, reply, resHeaders):
        self.client_address = (
            '', )  # avoid exception in BaseHTTPServer.py log_message()
        self.send_response(code)

        # We may need more fixes here, this kind of header is just for content-type
        if len(resHeaders) > 0:
            for h in resHeaders:
                self.send_header('Content-type', h)
                self.send_header('Connection', 'close')

        # self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(reply)


# ----------------------#


def start():
    global sockAddr
    load_dotenv()
    sockAddr = os.getenv("Proxy_URL")

    # Make sure the socket does not already exist
    try:
        print(f"Removing old socket file at {sockAddr}")
        os.unlink(sockAddr)
    except OSError as e:
        if os.path.exists(sockAddr):
            print(f"Failed to remove old socket file: {e}")
            raise

    unixSock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unixSock.settimeout(20)

    print('Binding on %s' % sockAddr)
    unixSock.bind(sockAddr)

    # Listen for incoming connections
    unixSock.listen(5)

    server = ThreadingHTTPServer(sockAddr, HTTPHandler, False)

    # ThreadingHTTPServer
    server.socket = unixSock
    server.serve_forever()

    # Cleanup after server stops
    try:
        unixSock.shutdown(socket.SHUT_RDWR)
        unixSock.close()
        os.remove(sockAddr)
        print(f"HTTP server stopped and socket cleaned up.")
    except Exception as e:
        print(f"Error during socket cleanup: {e}")

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