# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
The example notebooks and the test suite, installed with the package.

The package installs its example notebooks (the ``examples`` folder of the
repository) and its test suite (the ``tests`` folder), so that they always
match the installed version.  They are installed inside the package, where
they should not be run or changed: :func:`copy_examples` copies them to a
folder of your choice.  From a terminal::

    python -m vertex_algebras.examples ~/vertex-algebra-notebooks

The folder (any path, which must not exist yet) then holds the notebooks,
with their ``README.md``, the folder ``tests`` and the licence.  Start
Jupyter there with ``sage -n jupyter``.  ``bash tests/platform_report.sh``
there runs the whole test suite of the installed package.

EXAMPLES::

    sage: import os, tempfile
    sage: from vertex_algebras.examples import copy_examples
    sage: folder = os.path.join(tempfile.mkdtemp(), 'notebooks')
    sage: copy_examples(folder) == folder
    True
    sage: names = sorted(os.listdir(folder))
    sage: names[:2], names[-4:]
    (['01_getting_started.ipynb', '02_lie_superalgebras.ipynb'],
     ['16_virasoro_c0_fusion.ipynb', 'LICENSE', 'README.md', 'tests'])

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import os
import shutil
import sys

#: What is left out of the copy: caches and files written by earlier runs.
_IGNORE = shutil.ignore_patterns('triplet_data', '.ipynb_checkpoints',
                                 '__pycache__', 'platform_report_*.txt')


def _source_root():
    r"""
    The folder that holds ``examples``, ``tests`` and ``LICENSE``.

    An installed package has them in its subfolder ``_examples``; an
    editable install (``pip install -e .``) runs from the repository,
    whose root has them.

    TESTS::

        sage: import os
        sage: from vertex_algebras.examples import _source_root
        sage: root = _source_root()
        sage: all(os.path.exists(os.path.join(root, f))
        ....:     for f in ['examples', 'tests', 'LICENSE'])
        True
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.join(here, '_examples'),
                 os.path.dirname(os.path.dirname(here))):
        if (os.path.isdir(os.path.join(root, 'examples'))
                and os.path.isdir(os.path.join(root, 'tests'))):
            return root
    raise FileNotFoundError('the example notebooks are not installed with '
                            'vertex_algebras (looked in {})'.format(here))


def copy_examples(folder):
    r"""
    Copy the example notebooks and the test suite to ``folder``.

    INPUT:

    - ``folder`` -- a path (a string; ``~`` means the home folder) that
      does not exist yet; it is created

    OUTPUT:

    The path of the folder.  It holds the notebooks, their ``README.md``
    and the ``LICENSE``, and the test suite in the subfolder ``tests``
    (``bash tests/platform_report.sh`` runs it).

    EXAMPLES::

        sage: import os, tempfile
        sage: from vertex_algebras.examples import copy_examples
        sage: folder = os.path.join(tempfile.mkdtemp(), 'notebooks')
        sage: _ = copy_examples(folder)
        sage: len([f for f in os.listdir(folder) if f.endswith('.ipynb')])
        16
        sage: sorted(os.listdir(os.path.join(folder, 'tests')))[:3]
        ['axioms.py', 'crosscheck_lattice_engine.sage', 'crosscheck_module_engine.sage']

    TESTS:

    An existing folder is never overwritten::

        sage: copy_examples(folder)
        Traceback (most recent call last):
        ...
        FileExistsError: ... exists already: choose a new folder
    """
    folder = os.path.abspath(os.path.expanduser(str(folder)))
    if os.path.exists(folder):
        raise FileExistsError('{} exists already: choose a new folder'
                              .format(folder))
    root = _source_root()
    shutil.copytree(os.path.join(root, 'examples'), folder, ignore=_IGNORE)
    shutil.copytree(os.path.join(root, 'tests'),
                    os.path.join(folder, 'tests'), ignore=_IGNORE)
    shutil.copy2(os.path.join(root, 'LICENSE'), folder)
    return folder


def _main(argv):
    r"""
    The command ``python -m vertex_algebras.examples FOLDER``.

    Returns the exit status.

    TESTS::

        sage: import os, tempfile
        sage: from vertex_algebras.examples import _main
        sage: folder = os.path.join(tempfile.mkdtemp(), 'nb')
        sage: _main([folder])
        Copied 16 example notebooks and the test suite to .../nb
        Start Jupyter there with:
            cd .../nb
            sage -n jupyter
        0
        sage: _main([folder])
        .../nb exists already: choose a new folder
        1
        sage: _main([])
        usage: python -m vertex_algebras.examples FOLDER
        ...
        2
    """
    if len(argv) != 1 or argv[0].startswith('-'):
        print('usage: python -m vertex_algebras.examples FOLDER\n'
              'Copies the example notebooks and the test suite of '
              'vertex_algebras to FOLDER\n(any path that does not exist '
              'yet), e.g. ~/vertex-algebra-notebooks')
        return 2
    try:
        folder = copy_examples(argv[0])
    except (FileExistsError, FileNotFoundError, OSError) as err:
        print(err)
        return 1
    n = len([f for f in os.listdir(folder) if f.endswith('.ipynb')])
    print('Copied {} example notebooks and the test suite to {}\n'
          'Start Jupyter there with:\n'
          '    cd {}\n'
          '    sage -n jupyter'.format(n, folder, folder))
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
