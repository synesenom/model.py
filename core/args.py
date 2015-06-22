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
    """
    Class that manages arguments.
    More or less just a layer between argparse and the script.
    """
    def __init__(self, name="", desc=""):
        """
        Initializer.

        :param name: name of the script.
        :param desc: description of th script.
        """
        self._parser = argparse.ArgumentParser(prog=name, description=desc)
        self._arguments = list()

    def add(self, key, type=str, required=False, nargs=None, action="store", dest="", default=0, help=""):
        """
        Adds an argument. It returns the object itself in order to chain arguments
        together.

        :param key: key for argument.
        :param type: type of argument.
        :param required: is it required?
        :param nargs: number of arguments.
        :param action: what to do with the argument?
        :param dest: destination variable.
        :param default: default value.
        :param help: content of help menu.
        :return: the object itself.
        """
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
        """
        Prints argument values and returns the parameters parsed from the parser.

        :return: parameters in a dictionary.
        """
        _args = vars(self._parser.parse_args())
        print("")
        for _a in self._arguments:
            print("%s: %s" % (_a, _args[_a]))
        print("")
        return _args
