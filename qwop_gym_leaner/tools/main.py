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
import argparse
import os
import sys

import yaml

from . import common

__all__: list[str] = ["main"]


def run(action, cfg, tag=None):
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})
    expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
    common.register_env(expanded_env_kwargs, env_wrappers)

    match action:
        case "benchmark":
            from .benchmark import benchmark

            benchmark(steps=cfg.get("steps", 10000))

        case _:
            print("Unknown action: %s" % action)


def ensure_bootstrapped(progname):
    if os.path.isdir("config"):
        return

    print(
        f"""
This looks like a first-time run.
To perform initial setup, please run command this in your terminal:

    {progname} bootstrap
"""
    )

    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS)
    parser.add_argument("extra", help=argparse.SUPPRESS, nargs="?")
    parser.add_argument(
        "-c",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="config file, defaults to config/<action>.yml",
    )

    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    parser.usage = "%(prog)s [options] <action>"
    parser.epilog = """
action:
  benchmark         evaluate the actions/s achievable with this env
  bootstrap         perform initial setup
  help              print this help message

examples:
  %(prog)s benchmark
  %(prog)s bootstrap
"""

    args = parser.parse_args()

    # bootstrap does not use config files
    if args.action == "bootstrap":
        from .bootstrap import bootstrap

        bootstrap()
    else:
        ensure_bootstrapped(parser.prog)

        if args.c is None:
            args.c = open(os.path.join("config", f"{args.action}.yml"), "r")

        print("Loading configuration from %s" % args.c.name)
        cfg = yaml.safe_load(args.c)
        args.c.close()

        run(args.action, cfg)


if __name__ == "__main__":
    main()
