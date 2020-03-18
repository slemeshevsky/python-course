# -*- coding: utf-8 -*-

import sys

#Start 1st block
Language = "en"

ENGLISH = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
           5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
RUSSIAN = {0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре",
           5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять"}
#End 1st block

def main():
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print("usage: {0} [en|ru] number".format(sys.argv[0]))
        sys.exit()

    args = sys.argv[1:]
    if args[0] in {"en", "ru"}:
        global Language
        Language = args.pop(0)
    print_digits(args.pop(0))


def print_digits(digits):
    dictionary = ENGLISH if Language == "en" else RUSSIAN
    for digit in digits:
        print(dictionary[int(digit)], end=" ")
    print()


main()
