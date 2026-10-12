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
import contextlib
import struct

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync import client

from .log import Log
from .wsproto import ProtocolError
from .wsproto import to_bytes
from .wsproto import WSProto
from .wsserver import BROWSER_TIMEOUT

__all__: list[str] = ["STEP_TIMEOUT", "WSClient", "WSClientMock"]

STEP_TIMEOUT = 10  # seconds to wait for the reply to a command


class WSClient:
    """Synchronous client QwopEnv uses to exchange messages with its relay (`WSServer`).

    Failures are never retried: a command that may or may not have reached the page
    cannot be safely re-sent. Any failed exchange closes the connection for good (see
    `poison`), so a late reply can never be read as the answer to another command.
    """

    def __init__(self, server, loglevel):
        self.server = server
        self.logger = Log.get_logger(__name__, loglevel)
        self._stack = contextlib.ExitStack()
        self._broken = None  # why the connection was closed for good (a message, not the exception)

        try:
            self.ws = self._stack.enter_context(
                client.connect(f"ws://127.0.0.1:{server.port}", open_timeout=10, compression=None)
            )
        except (OSError, TimeoutError) as e:
            raise self._failure("could not connect to the QWOP relay", e) from (server.error or e)

        try:
            # The relay acknowledges only once the page has loaded and registered
            reply = self.send(to_bytes(WSProto.H_REG) + to_bytes(WSProto.REG_PY), timeout=BROWSER_TIMEOUT + 5)
            if reply[0] != WSProto.H_ACK:
                raise ProtocolError(f"expected an ACK header, got: {reply[0]}")
        except BaseException:
            self.close()
            raise
        self.logger.debug("Connected")

    def _failure(self, msg, cause):
        if self.server.error is not None:
            return ConnectionError(
                f"{msg}: the relay failed with {type(self.server.error).__name__}: {self.server.error}"
            )
        return ConnectionError(f"{msg}: {type(cause).__name__}: {cause}")

    def poison(self, error):
        """Close the connection for good; every later command raises, naming `error`."""
        if self._broken is None:
            # only the text is kept: the exception's traceback would keep the env alive
            self._broken = f"{type(error).__name__}: {error}"
        with contextlib.suppress(Exception):
            self.close()

    def send(self, data, timeout=STEP_TIMEOUT):
        """Send one message and return the reply."""
        if self._broken is not None:
            raise ConnectionError(
                f"the connection to the QWOP page was closed after an earlier failure ({self._broken})"
            )
        try:
            self.ws.send(data)
            reply = self.ws.recv(timeout=timeout)
        except TimeoutError as e:
            error = self._failure(f"no reply from the QWOP page within {timeout} s", e)
            self.poison(error)
            raise error from (self.server.error or e)
        except (ConnectionClosed, OSError) as e:
            error = self._failure("lost the connection to the QWOP page", e)
            self.poison(error)
            raise error from (self.server.error or e)
        except BaseException as e:  # e.g. KeyboardInterrupt while waiting for the reply
            self.poison(e)
            raise

        if reply[0] == WSProto.H_ERR:
            # the page failed half-way through the command: its state is unknown
            error = RuntimeError(f"the QWOP page raised an error:{reply[1:].decode(errors='replace')}")
            self.poison(error)
            raise error
        return reply

    def close(self):
        self._stack.close()


class WSClientMock:
    """Stand-in for `WSClient` that needs no browser: ACKs reloads, answers anything else with a fixed observation."""

    def send(self, data, timeout=None):
        if data[0] == WSProto.H_RLD:
            return to_bytes(WSProto.H_ACK)
        return (
            to_bytes(WSProto.H_OBS)
            + to_bytes(0)  # flags
            + struct.pack("=f", 2 / 300)  # time: two frames, as right after a reset
            + struct.pack("=f", 0.25)  # distance: the starting position
            + np.zeros(60, dtype=np.float32).tobytes()  # obs
        )

    def poison(self, error):
        pass

    def close(self):
        pass
