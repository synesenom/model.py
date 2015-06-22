#!/usr/bin/env python3
#title          : utils.py
#description    : Contains convenience methods.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python utils.py
#=====================================================
import csv


def read_csv(filename):
    """
    Reads data from a csv file.
    Note: first line is reserved for header, so it is ignored.

    :param filename: name of the data file.
    :return: data in a list structure.
    """
    with open(filename, 'r') as _input_file:
        return list(list(row) for row in csv.reader(_input_file, delimiter=' '))[1:]


def print_csv(filename, header, data):
    """
    Prints out data in a csv file with given header.

    :param filename: name of the output file.
    :param header:   list of column names.
    :param data:     data to print out.
    """
    with open(filename, 'w') as _output_file:
        csv_out = csv.writer(_output_file, delimiter=' ', quotechar='"')
        csv_out.writerow(['#'] + header)
        for row in data:
            if type(row) is tuple or type(row) is list:
                csv_out.writerow(row)
            else:
                csv_out.writerow([row])
