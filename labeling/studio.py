"""Authenticated HTTP client for Label Studio session requests."""

from http.cookiejar import CookieJar
import json
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener


class Studio:
    def __init__(self, base):
        self.base = base
        self.cookies = CookieJar()
        self.client = build_opener(ProxyHandler({}), HTTPCookieProcessor(self.cookies))

    def request(self, path, data=None, *, form=False):
        headers = {"Referer": self.base + path}
        if data is not None:
            csrf = next((cookie.value for cookie in self.cookies if cookie.name == "csrftoken"), None)
            if csrf is None:
                raise RuntimeError("Label Studio did not issue a CSRF cookie")
            headers["X-CSRFToken"] = csrf
            headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
            data = (urlencode(data) if form else json.dumps(data)).encode()
        with self.client.open(Request(self.base + path, data=data, headers=headers), timeout=10) as response:
            return response.status, response.read()
