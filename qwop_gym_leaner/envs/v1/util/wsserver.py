# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.
#           2025 Emanuele Ballarin <emanuele@ballarin.cc>
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import asyncio
import contextlib
import logging
import pathlib
import sys
import threading
import urllib.parse
import uuid

import websockets
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from websockets.exceptions import ConnectionClosed

from .log import Log
from .wsproto import ProtocolError
from .wsproto import to_bytes
from .wsproto import WSProto

__all__: list[str] = ["BROWSER_TIMEOUT", "RELOAD_TIMEOUT", "WSServer"]

BROWSER_TIMEOUT = 60  # seconds for the browser to launch, load the page and register
RELOAD_TIMEOUT = 30  # seconds for a page reload to complete and the page to re-register


class _Stopped(Exception):
    """A requested shutdown interrupted a wait inside the relay."""


class Peer:
    """One side of the relay: the QWOP page ("js") or QwopEnv ("py")."""

    def __init__(self, name):
        self.ws = None
        self.other = None
        self.name = name


class WSServer:
    """Relay between the QWOP page and QwopEnv, running in a daemon thread of the env's process.

    The thread owns an asyncio event loop, the websocket server and the selenium-driven
    browser. The first fatal error is stored in `error` before any connection is closed,
    so that `WSClient` can re-raise it in the env's thread.
    """

    def __init__(
        self,
        sock,
        seed,
        driver,
        browser,
        stepsize,
        stat_in_browser,
        text_in_browser,
        game_in_browser,
        loglevel,
    ):
        seedmin = -9007199254740991  # js Number.MIN_SAFE_INTEGER
        seedmax = 9007199254740991  # js Number.MAX_SAFE_INTEGER
        if not seedmin <= seed <= seedmax:
            raise ValueError(f"seed must be between {seedmin} and {seedmax}, got: {seed}")

        self.sock = sock
        self.port = sock.getsockname()[1]
        self.seed = seed
        self.stat_in_browser = stat_in_browser
        self.game_in_browser = game_in_browser
        self.text_in_browser = text_in_browser
        self.driver = driver
        self.browser = browser
        self.stepsize = stepsize
        self.logger = Log.get_logger(__name__, loglevel)

        self.error = None
        self.shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"qwop-wsserver-{self.port}", daemon=True)
        self._jspeer = Peer("js")
        self._pypeer = Peer("py")
        self._jspeer.other = self._pypeer
        self._pypeer.other = self._jspeer
        self._peers = {}
        self._driver = None
        self._ready = None  # asyncio.Event set when the page has registered; created on the server's loop

    def start(self):
        self._thread.start()

    def stop(self, timeout=30):
        """Signal shutdown and wait (up to `timeout` seconds) for the thread to quit the browser."""
        self.shutdown.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def is_alive(self):
        return self._thread.is_alive()

    def _fail(self, error):
        if isinstance(error, _Stopped):
            return
        if self.error is None:
            self.error = error
            self.logger.error("%s: %s", type(error).__name__, error)
        self.shutdown.set()

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # noqa: BLE001 - any failure must reach the env via `self.error`
            self._fail(e)
        finally:
            self.shutdown.set()
            self._quit_browser()

    async def _main(self):
        self._ready = asyncio.Event()
        with self.sock:
            # small binary messages: compression only adds latency
            async with websockets.serve(self.handler, sock=self.sock, compression=None):
                self.logger.info("Listening on port %d", self.port)
                try:
                    await asyncio.to_thread(self._launch_browser)
                    await self._wait_ready(BROWSER_TIMEOUT, "startup")
                    while not self.shutdown.is_set():
                        await asyncio.sleep(0.1)
                except Exception as e:
                    self._fail(e)  # record it before the connections close
                    raise
        self.logger.info("Shut down")

    async def _wait_ready(self, timeout, what):
        """Wait for the page to register, giving up on shutdown (the server's close waits for every handler)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._ready.is_set():
            if self.shutdown.is_set():
                raise _Stopped(what)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(f"browser page did not connect within {timeout} s ({what})")
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(min(0.1, remaining)):
                    await self._ready.wait()

    def build_url(self):
        file = pathlib.Path(__file__).parents[1].joinpath("game", "QWOP.html")
        text = urllib.parse.quote_plus(self.text_in_browser or "")
        return (
            f"file://{file}"
            f"?port={self.port:d}"
            f"&seed={self.seed:d}"
            f"&game={self.game_in_browser:d}"
            f"&stat={self.stat_in_browser:d}"
            f"&text={text}"
            f"&intro=0"
            f"&stepsize={self.stepsize:d}"
        )

    def _launch_browser(self):
        """Start Chrome and load the game (blocking; runs in a worker thread)."""
        self.logger.info("Launching web browser...")

        options = webdriver.ChromeOptions()
        options.add_argument("allow-file-access-from-files")
        options.add_argument("allow-cross-origin-auth-prompt")
        options.add_argument(f"user-agent=Chrome-{uuid.uuid4()}")
        options.add_argument("disable-infobars")
        options.add_argument("disable-extensions")
        options.add_argument("disable-popup-blocking")
        options.add_argument("disable-notifications")

        # Headless mode for when the game is not displayed
        # Based on: https://developer.chrome.com/blog/supercharge-web-ai-testing
        if not self.game_in_browser:
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--enable-gpu")
            options.add_argument("--use-gl=angle")
            options.add_argument("--use-angle=swiftshader")

        if self.game_in_browser and (self.stat_in_browser or self.text_in_browser):
            options.add_argument("window-size=1160,585")
            options.add_argument("window-position=650,130")
        else:
            options.add_argument("window-size=660,585")
            options.add_argument("window-position=650,130")

        # https://www.selenium.dev/documentation/webdriver/getting_started/upgrade_to_selenium_4/#python-1
        service = ChromeService(executable_path=self.driver)

        # https://community.brave.com/t/is-there-a-selenium-driver-for-brave-browser/49696/4
        options.binary_location = self.browser
        options.add_argument("--incognito")

        self._driver = webdriver.Chrome(service=service, options=options)
        self._driver.get(self.build_url())

    def _quit_browser(self):
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception as e:  # noqa: BLE001 - best effort while shutting down
                self.logger.warning("Failed to quit the browser: %s", e)
            self._driver = None

    async def _register_peer(self, ws, peer_id):
        match peer_id:
            case WSProto.REG_JS:
                peer = self._jspeer
            case WSProto.REG_PY:
                peer = self._pypeer
            case _:
                raise ProtocolError(f"unknown peer id: {peer_id}")

        if peer.ws is not None and peer.ws is not ws:
            await peer.ws.close()

        peer.ws = ws
        self._peers[ws] = peer

        if peer is self._pypeer:
            self.logger.info("Waiting for browser...")
            await self._wait_ready(BROWSER_TIMEOUT, "startup")

        await self.send(peer, to_bytes(WSProto.H_ACK))
        self.logger.info("%s client registration ACK", peer.name)

        if peer is self._jspeer:
            self._ready.set()

    async def _reload(self, src_peer, seed):
        if src_peer is not self._pypeer:
            raise ProtocolError(f"received RELOAD from the {src_peer.name} peer")
        self.logger.info("Reloading browser page with new seed: %s", seed)

        self.seed = seed
        ws, self._jspeer.ws = self._jspeer.ws, None
        self._ready.clear()

        if ws is not None:
            await ws.close()
        await asyncio.to_thread(self._driver.get, self.build_url())
        await self._wait_ready(RELOAD_TIMEOUT, "reload")
        await self.send(self._pypeer, to_bytes(WSProto.H_ACK))

    async def handler(self, ws):
        try:
            await self._handler(ws)
        except Exception as e:  # noqa: BLE001 - a broken peer is fatal: the env cannot continue
            self._fail(e)
        finally:
            peer = self._peers.pop(ws, None)
            if peer is not None and peer.ws is ws:
                peer.ws = None

    async def _handler(self, ws):
        self.logger.info("Connection from: %s", ws.request.headers.get("user-agent"))

        async for data in ws:
            src = self._peers.get(ws)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(Log.format_inbound(data, src))

            header = data[0]
            if src is None and header != WSProto.H_REG:
                raise ProtocolError(f"unregistered peer sent header {header}")

            match header:
                # put most common match cases on top
                case WSProto.H_OBS | WSProto.H_CMD:
                    try:
                        await self.send(src.other, data)
                    except (ProtocolError, ConnectionClosed) as e:
                        if src is not self._jspeer:
                            raise
                        # the env gave up on this reply (timeout, interrupt) and closed its connection
                        self.logger.debug("Dropped a reply for the disconnected env: %s", e)
                case WSProto.H_REG:
                    await self._register_peer(ws, data[1])
                case WSProto.H_RLD:
                    await self._reload(src, int.from_bytes(data[1:5], sys.byteorder))
                case WSProto.H_LOG:
                    longfmt = self.logger.level == logging.DEBUG
                    self.logger.info(Log.format_remote(data[1:], src, longfmt))
                case WSProto.H_ERR:
                    # forwarded so that the env raises it instead of waiting for a reply
                    message = data[1:].decode(errors="replace")
                    self.logger.error("JS error: %s", message)
                    if self._pypeer.ws is None:
                        raise RuntimeError(f"the QWOP page raised an error:{message}")
                    await self.send(self._pypeer, data)
                case _:
                    raise ProtocolError(f"unknown header {header} from the {src.name} peer")

    async def send(self, peer, data):
        if peer.ws is None:
            raise ProtocolError(f"the {peer.name} peer is not connected")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(Log.format_outbound(data, peer))
        await peer.ws.send(data)
