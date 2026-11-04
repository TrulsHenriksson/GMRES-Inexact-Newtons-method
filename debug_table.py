from typing import Any


def debug_table_header(*headers: tuple[str, str], indent=""):
    strings = [f"{header:{fmt}}" for header, fmt in headers]
    separators = ["━" * len(string) for string in strings]
    print(indent + strings[0], *strings[1:], sep=" ┃ ", flush=True)
    print(indent + separators[0], *separators[1:], sep="━╋━", flush=True)

def debug_table_row(*values: tuple[Any, str], indent=""):
    first, *rest = [f"{value:{fmt}}" for value, fmt in values]
    print(indent + first, *rest, sep=" ┃ ", flush=True)
