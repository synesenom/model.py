#!/usr/bin/env python3
#title          : args.py
#description    : Argument handler class.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python args.py
#=====================================================
import argparse


class Args:
    def __init__(self, name="", desc="", show_settings=False):
        self._parser = argparse.ArgumentParser(prog=name, description=desc)
        self._show_settings = show_settings
        self._arguments = list()

    def add(self, key, type=str, required=False, nargs=None, action="store", dest="", default=0, help=""):
        if action == "store":
            self._parser.add_argument(key,
                                      type=type,
                                      required=required,
                                      action=action,
                                      dest=dest,
                                      nargs=nargs,
                                      default=default,
                                      help=help)
        else:
            self._parser.add_argument(key,
                                      required=required,
                                      action=action,
                                      dest=dest,
                                      help=help)
        self._arguments.append(dest)
        return self

    def get(self):
        _args = vars(self._parser.parse_args())
        if self._show_settings:
            print("")
            for _a in self._arguments:
                print("%s: %s" % (_a, _args[_a]))
            print("")
        return _args
