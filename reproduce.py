"""One entry point for saved-result reproduction and the finite-state simulation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'restore-sources': ['-m', 'tools.reproduce.restore_sources'],
    'figures': ['-m', 'tools.reproduce.figures'],
    'cell-plots': ['-m', 'tools.reproduce.cell_plots'],
    'raid-tables': ['-m', 'tools.reproduce.raid'],
    'simulation': ['simulation/reproduce.py'],
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Options after the command are forwarded unchanged. Use COMMAND --help '
        'for details. Relative paths resolve from the repository root.')
    parser.add_argument('command', choices=['verify', *COMMANDS])
    command = parser.parse_args(args[:1]).command
    forwarded = args[1:]
    if command == 'verify':
        if forwarded:
            parser.error('verify checks all included reference inventories and takes no options')
        steps = [['-m', 'tools.reproduce.verify'], ['simulation/reproduce.py', 'verify']]
    else:
        steps = [[*COMMANDS[command], *forwarded]]
    for step in steps:
        result = subprocess.run([sys.executable, '-B', *step], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
