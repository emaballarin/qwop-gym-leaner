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
    def format(self, record):
        record.reltime = f"{record.relativeCreated / 1000:.2f}"
        return super().format(record)


class Log:
    # 0 = print ALL messages
    # 1 = print a (short) version of ALL messages
    # 2 = filter out in/outbound messages, print the rest
    # 3 = print only the explicit LOG calls
    # 4 = silence

    LEVEL = 4
    DELIM_OUT = ">" * 79
    DELIM_IN = "<" * 79
    DELIM_REMOTE = ":" * 79

    # for logging purposes
    HMAP: ClassVar[dict] = {
        WSProto.H_REG: "H_REG",
        WSProto.H_ACK: "H_ACK",
        WSProto.H_REJ: "H_REJ",
        WSProto.H_CMD: "H_CMD",
        WSProto.H_OBS: "H_OBS",
        WSProto.H_IMG: "H_IMG",
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
        WSProto.CMD_IMG: "IMG",
        WSProto.CMD_DRW: "DRW",
    }

    OBSMAP: ClassVar[dict] = {
        WSProto.OBS_PAS: "PAS",
        WSProto.OBS_END: "END",
        WSProto.OBS_SUC: "SUC",
    }

    IMGMAP: ClassVar[dict] = {
        WSProto.IMG_JPG: "JPG",
        WSProto.IMG_PNG: "PNG",
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

        if direction == "in":
            arrow = "<--"
        else:
            arrow = "-->"

        header_uint = data[0]
        header_name = Log.HMAP.get(header_uint)
        header_repr = f"{header_uint:08b}"

        if header_uint == WSProto.H_REG:
            header2_uint = data[1]
            header2_name = Log.REGMAP.get(header2_uint)
            header2_repr = f"{header2_uint:08b}"

            line2 = f"{header_name!s:<16} | {header2_name!s:<16} |"
            line3 = f"{header_repr:<16} | {header2_repr:<16} | {Log.data_repr(data[2:])}"
        elif header_uint == WSProto.H_CMD:
            header2_uint = data[1]
            names = [v for k, v in Log.CMDMAP.items() if header2_uint & k]

            header2_name = "+".join(names)
            header2_repr = f"{header2_uint:08b}"

            line2 = f"{header_name!s:<16} | {header2_name!s:<16} | (data)"
            line3 = f"{header_repr:<16} | {header2_repr:<16} | {Log.data_repr(data[2:])}"
        elif header_uint == WSProto.H_OBS:
            header2_uint = data[1]
            names = [v for k, v in Log.OBSMAP.items() if header2_uint & k]

            header2_name = "+".join(names)
            header2_repr = f"{header2_uint:08b}"

            line2 = f"{header_name!s:<16} | {header2_name!s:<16} | {'(data)':<16}"
            line3 = f"{header_repr:<16} | {header2_repr:<16} | {Log.data_repr(data[2:]):<16}"
        elif header_uint == WSProto.H_IMG:
            header2_uint = data[1]
            header2_name = Log.IMGMAP.get(header2_uint)
            header2_repr = f"{header2_uint:08b}"

            line2 = f"{header_name!s:<16} | {header2_name!s:<16} | {'(data)':<16}"
            line3 = f"{header_repr:<16} | {header2_repr:<16} | {Log.data_repr(data[2:]):<16}"
        else:
            line2 = f"{header_name!s:<16} | {'(data)':<16}"
            line3 = f"{header_repr:<16} | {Log.data_repr(data[1:]):<16}"

        delim = "-" * 79

        if Log.LEVEL == 0:
            return f"{delim}\n[{arrow}{client_name}] {line2}\n[{arrow}{client_name}] {line3}"
        return f"[{arrow}{client_name}] {line2}"

    @staticmethod
    def data_repr(data):
        x = data.hex()
        return x if len(x) <= 16 else (x[:13] + "...")
