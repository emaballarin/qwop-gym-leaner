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
import sys

__all__: list[str] = ["ProtocolError", "WSProto", "to_bytes"]


def to_bytes(number, nbytes=1):
    """Encode a non-negative integer in native byte order."""
    return number.to_bytes(nbytes, sys.byteorder)


class ProtocolError(RuntimeError):
    """A peer sent a message that violates the wire protocol."""


class WSProto:
    """Wire-protocol constants shared with `game/ws.js` (keep both in sync)."""

    #
    # Header (uint8)
    #

    H_REG = 0  # reg req    (**->**) payload: id (uint8)
    H_ACK = 1  # reg ack    (**->**)
    H_CMD = 3  # cmd        (py->js) payload: cmdflags (uint8) + step (uint16) + rew (float32) + tot_rew (float32)
    H_OBS = 4  # obs        (js->py) payload: flags (uint8) + time (float32) + distance (float32) + obs ([60]float32)
    H_LOG = 6  # log        (js->py) payload: msg (utf-8)
    H_ERR = 7  # error      (js->py) payload: msg (utf-8)
    H_RLD = 8  # reload     (py->srv) payload: seed (uint32)

    #
    # Data
    #

    # REG payload: id (uint8)
    REG_JS = 0  # js client
    REG_PY = 1  # py client

    # CMD payload: cmdflags (uint8)
    CMD_STP = 0b00000001  # advance 1 timestep
    CMD_K_Q = 0b00000010  # key Q
    CMD_K_W = 0b00000100  # key W
    CMD_K_O = 0b00001000  # key O
    CMD_K_P = 0b00010000  # key P
    CMD_RST = 0b00100000  # restart game
    CMD_DRW = 0b10000000  # draw (browser render)

    # OBS payload: flags (uint8)
    OBS_END = 0b00000010  # game has ended
    OBS_SUC = 0b00000100  # run was successful (100+m)
