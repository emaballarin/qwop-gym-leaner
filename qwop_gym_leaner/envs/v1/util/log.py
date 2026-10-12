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
import logging
from typing import ClassVar

from .wsproto import WSProto

__all__: list[str] = ["Log"]


class RelativeTimeFormatter(logging.Formatter):
    """Formatter exposing the time since logging start as `%(reltime)s`."""

    def format(self, record):
        record.reltime = f"{record.relativeCreated / 1000:.2f}"
        return super().format(record)


class Log:
    """Logger factory and one-line formatters for wire-protocol messages."""

    DELIM_REMOTE = ":" * 79

    HMAP: ClassVar[dict] = {
        WSProto.H_REG: "H_REG",
        WSProto.H_ACK: "H_ACK",
        WSProto.H_CMD: "H_CMD",
        WSProto.H_OBS: "H_OBS",
        WSProto.H_LOG: "H_LOG",
        WSProto.H_ERR: "H_ERR",
        WSProto.H_RLD: "H_RLD",
    }

    REGMAP: ClassVar[dict] = {
        WSProto.REG_JS: "REG_JS",
        WSProto.REG_PY: "REG_PY",
    }

    CMDMAP: ClassVar[dict] = {
        WSProto.CMD_STP: "STP",
        WSProto.CMD_K_Q: "K_Q",
        WSProto.CMD_K_W: "K_W",
        WSProto.CMD_K_O: "K_O",
        WSProto.CMD_K_P: "K_P",
        WSProto.CMD_RST: "RST",
        WSProto.CMD_DRW: "DRW",
    }

    OBSMAP: ClassVar[dict] = {
        WSProto.OBS_END: "END",
        WSProto.OBS_SUC: "SUC",
    }

    @staticmethod
    def get_logger(name, level):
        logger = logging.getLogger(name.split(".")[-1])
        logger.setLevel(getattr(logging, level))

        # Avoid stacking duplicate handlers when called repeatedly for the same
        # logger name (e.g. one QwopEnv per worker in a vectorized environment),
        # which would otherwise print every log line once per handler.
        if not logger.handlers:
            fmt = "-- %(reltime)ss [%(name)s] %(levelname)s: %(message)s"
            formatter = RelativeTimeFormatter(fmt)

            loghandler = logging.StreamHandler()
            loghandler.setLevel(logging.DEBUG)
            loghandler.setFormatter(formatter)
            logger.addHandler(loghandler)

        return logger

    @staticmethod
    def format_remote(msg, client, longfmt):
        client_name = client.name if client else "??"
        decoded = msg.decode("utf-8")

        if longfmt:
            return f"{Log.DELIM_REMOTE}\n[ {client_name} ] {decoded}"
        return f"[ {client_name} ] {decoded}"

    @staticmethod
    def format_outbound(data, client):
        return Log.format_allbound(data, client, "out")

    @staticmethod
    def format_inbound(data, client):
        return Log.format_allbound(data, client, "in")

    @staticmethod
    def format_allbound(data, client, direction):
        client_name = client.name if client else "??"
        arrow = "<--" if direction == "in" else "-->"
        header_name = Log.HMAP.get(data[0])

        if data[0] == WSProto.H_REG:
            detail = f"{Log.REGMAP.get(data[1])!s:<16} |"
        elif data[0] == WSProto.H_CMD:
            detail = f"{'+'.join(v for k, v in Log.CMDMAP.items() if data[1] & k)!s:<16} | (data)"
        elif data[0] == WSProto.H_OBS:
            detail = f"{'+'.join(v for k, v in Log.OBSMAP.items() if data[1] & k)!s:<16} | {'(data)':<16}"
        else:
            detail = f"{'(data)':<16}"

        return f"[{arrow}{client_name}] {header_name!s:<16} | {detail}"
