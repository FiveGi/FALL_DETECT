# -*- coding: utf-8 -*-
"""A throwaway MJPEG stream, so the reconnect path can be tested by killing it.

The camera loop treats anything that is not a file as a network stream, and until now a single
failed read ended the loop for good. Testing that needs a stream that can actually be taken
away and given back, which is what this is: it serves Test/13.mp4 on a loop as MJPEG over HTTP.
"""
import http.server
import socketserver
import threading
import time

import cv2

PORT = 8090
CLIP = '/app/Test/13.mp4'


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
        self.end_headers()
        cap = cv2.VideoCapture(CLIP)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                 + str(len(buf)).encode() + b'\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(1 / 24.0)
        except Exception:
            pass
        finally:
            cap.release()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


print('mjpeg on %d' % PORT, flush=True)
Server(('0.0.0.0', PORT), Handler).serve_forever()
