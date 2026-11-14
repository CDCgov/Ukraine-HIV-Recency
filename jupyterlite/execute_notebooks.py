"""Execute the notebooks in ``jupyterlite/content/`` in place.

    python jupyterlite/execute_notebooks.py [name ...]

Runs every ``0*.ipynb`` (or the names given) with the kernel of the Python that
runs this script, saves the outputs into the notebook, and puts back the
Pyodide kernelspec so the files still open with the in-browser kernel.
build_site.ps1 calls this with the project's build environment.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

CONTENT = Path(__file__).resolve().parent / 'content'


def main() -> int:
    names = sys.argv[1:] or sorted(p.name for p in CONTENT.glob('0*.ipynb'))
    failed = 0
    for name in names:
        path = CONTENT / name
        nb = nbformat.read(path, as_version=4)
        kernelspec = nb.metadata.get('kernelspec')
        try:
            NotebookClient(nb, timeout=1800, kernel_name='python3', record_timing=False,
                           resources={'metadata': {'path': str(CONTENT)}}).execute()
            status = 'OK'
        except Exception as exc:  # keep going so one failure does not hide the others
            status = f'FAILED: {type(exc).__name__}: {str(exc)[-300:]}'
            failed += 1
        if kernelspec:
            nb.metadata['kernelspec'] = kernelspec
        nb.metadata['language_info'] = {'name': 'python'}
        nbformat.write(nb, path)
        print(f'{status:8s} {name}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
