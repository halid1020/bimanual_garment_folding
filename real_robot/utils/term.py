"""Terminal colouring for the xArm test scripts.

Green for what passed, red for what failed or errored, yellow for warnings that
are not verdicts. Colour is emitted only when stdout is a real terminal, so piping
a run into a file or through ``grep`` still gives clean, greppable text -- the
verification commands in XARM_TESTING.md depend on that.

Honours the ``NO_COLOR`` convention (https://no-color.org) and ``TERM=dumb``.
"""
import os
import sys

_CODES = {
    'green': '\033[32m',
    'red': '\033[31m',
    'yellow': '\033[33m',
    'bold': '\033[1m',
}
_RESET = '\033[0m'


def colour_enabled(stream=None):
    stream = sys.stdout if stream is None else stream
    if os.environ.get('NO_COLOR') is not None:
        return False
    if os.environ.get('TERM', '') == 'dumb':
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _wrap(text, *names):
    if not colour_enabled():
        return text
    return ''.join(_CODES[n] for n in names) + text + _RESET


def green(text):
    return _wrap(text, 'green')


def red(text):
    return _wrap(text, 'red')


def yellow(text):
    return _wrap(text, 'yellow')


def bold(text):
    return _wrap(text, 'bold')


def verdict(ok, text=None, ok_text='PASS', bad_text='FAIL'):
    """Colour a pass/fail label: green when ``ok``, red otherwise."""
    label = text if text is not None else (ok_text if ok else bad_text)
    return green(label) if ok else red(label)


def mark(ok):
    """The inline waypoint marker used by the reach/dry-run reports."""
    return green('  ok ') if ok else red('  !! ')
