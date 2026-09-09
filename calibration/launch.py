"""Explicit runtime selection; no hidden dependency shims or model patches."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
args = sys.argv[1:]
for index, token in enumerate(args):
    if token == '--runtime-path':
        sys.path.append(str(Path(args[index+1]).resolve(strict=True)))
sys.path.insert(0, str(Path(args[args.index('--repo')+1]).resolve(strict=True)))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge import main
if __name__ == '__main__':
    main()
