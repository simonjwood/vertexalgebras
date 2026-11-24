#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
#
# Installs sage-vertex-algebras on Linux, macOS or Windows (WSL), with
# everything it needs:
#
#   1. on macOS, Apple's Command Line Tools (asks first);
#   2. Miniforge, a minimal conda, in ~/miniforge3 (asks first; an existing
#      Miniforge, Miniconda or Anaconda is used instead);
#   3. a conda environment "sage" with SageMath (asks first; an existing
#      one with Sage 10.10 or later is used instead);
#   4. the package, from PyPI, with pip;
#   5. a copy of the example notebooks in ~/vertex-algebra-notebooks
#      (asks first).
#
# Run it with:  bash install.sh   (bash install.sh --help lists the options)
# Running it again upgrades the package; it never changes or removes
# anything without asking.  Nothing here needs administrator rights, and
# your shell settings (.bashrc, .zshrc) are not changed.
#
# Written for bash 3.2 (macOS's bash): no associative arrays, no mapfile.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Please run this script with bash:  bash install.sh" >&2
    exit 1
fi

set -eEuo pipefail

# Keep in step with _MIN_SAGE in src/vertex_algebras/__init__.py.
MIN_SAGE_MAJOR=10
MIN_SAGE_MINOR=10
MIN_SAGE="$MIN_SAGE_MAJOR.$MIN_SAGE_MINOR"

PACKAGE=sage-vertex-algebras
PIP_SPEC=$PACKAGE
ISSUES=https://github.com/simonjwood/vertexalgebras/issues
MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download

ENV_NAME=sage
ALT_ENV_NAME=sage-vertex-algebras
CONDA_DIR=
NOTEBOOKS="$HOME/vertex-algebra-notebooks"
COPY_NOTEBOOKS=1
YES=0

usage() {
    cat <<EOF
Usage: bash install.sh [options]

Installs SageMath (with conda) and the package $PACKAGE.  Asks before
installing anything.

Options:
  --yes             answer yes to every question (for unattended installs)
  --env NAME        the conda environment to use or create (default: sage)
  --conda DIR       the conda installation to use, or where to install
                    Miniforge (default: an existing one, else ~/miniforge3)
  --notebooks DIR   where to copy the example notebooks
                    (default: ~/vertex-algebra-notebooks)
  --no-notebooks    do not copy the example notebooks
  --package SPEC    what pip installs (default: $PACKAGE); for testing,
                    e.g. a path to a copy of the repository
  -h, --help        show this help
EOF
}

die() {
    printf '\nError: %s\n' "$*" >&2
    exit 1
}

need_arg() {
    [ $# -ge 2 ] && [ -n "$2" ] || die "option $1 needs a value (see bash install.sh --help)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y) YES=1 ;;
        --env) need_arg "$@"; ENV_NAME=$2; shift ;;
        --conda) need_arg "$@"; CONDA_DIR=$2; shift ;;
        --notebooks) need_arg "$@"; NOTEBOOKS=$2; shift ;;
        --no-notebooks) COPY_NOTEBOOKS=0 ;;
        --package) need_arg "$@"; PIP_SPEC=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
done

# ~ in option values (bash leaves --opt=~/x alone, and quoted ~ too).
# shellcheck disable=SC2088  # a literal ~
expand_home() {
    case "$1" in
        "~") printf '%s\n' "$HOME" ;;
        "~/"*) printf '%s\n' "$HOME/${1#\~/}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}
# Paths under the home folder are shown as ~/...
# shellcheck disable=SC2088  # a literal ~
show_path() {
    case "$1" in
        "$HOME") printf '~\n' ;;
        "$HOME/"*) printf '~/%s\n' "${1#"$HOME"/}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

NOTEBOOKS=$(expand_home "$NOTEBOOKS")
[ -z "$CONDA_DIR" ] || CONDA_DIR=$(expand_home "$CONDA_DIR")

on_error() {
    printf '\nThe installation stopped because of the error above (line %s of install.sh).\n' "$1" >&2
    printf 'Running bash install.sh again continues where it stopped.  If the error\n' >&2
    printf 'comes back, please open an issue at\n    %s\n' "$ISSUES" >&2
    printf 'and attach the whole output of this script.\n' >&2
}
trap 'on_error $LINENO' ERR

say() {
    printf '\n==> %s\n' "$*"
}

# ask "question": 0 for yes.  Reads the answer from the terminal, also
# when the script itself comes through a pipe (curl ... | bash).
ask() {
    local answer=
    if [ "$YES" = 1 ]; then
        printf '\n%s [y/N] y (--yes)\n' "$1"
        return 0
    fi
    if [ -t 0 ]; then
        read -r -p $'\n'"$1 [y/N] " answer
    elif { : </dev/tty; } 2>/dev/null; then
        read -r -p $'\n'"$1 [y/N] " answer </dev/tty
    else
        die "there is no terminal to ask questions in; run bash install.sh in a terminal, or add --yes"
    fi
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# run command...: shows the command, then runs it.
run() {
    printf '\n    $ %s\n\n' "$*"
    "$@"
}

# ---------------------------------------------------------------------------
# 0. The system

OS=$(uname -s)
ARCH=$(uname -m)
WSL=0
case "$OS" in
    Linux)
        if grep -qi microsoft /proc/version 2>/dev/null; then WSL=1; fi
        case "$ARCH" in
            x86_64|aarch64) ;;
            *) die "SageMath is not available from conda-forge for Linux on $ARCH (only x86_64 and aarch64)." ;;
        esac
        ;;
    Darwin)
        # A Terminal running under Rosetta reports x86_64 on Apple Silicon:
        # install the native (arm64) conda anyway.
        if [ "$ARCH" = x86_64 ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
            ARCH=arm64
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        die "SageMath does not run on Windows itself.  Install the Windows Subsystem for Linux (WSL) and run this script in its Ubuntu window: see \"Quick install\" in the installation guide, https://github.com/simonjwood/vertexalgebras#quick-install"
        ;;
    *)
        die "this script works on Linux, macOS and Windows (WSL), not on $OS."
        ;;
esac

if [ "$WSL" = 1 ]; then
    SYSTEM="Windows (WSL), $ARCH"
elif [ "$OS" = Darwin ]; then
    SYSTEM="macOS $(sw_vers -productVersion 2>/dev/null || true), $ARCH"
else
    SYSTEM="Linux, $ARCH"
fi

cat <<EOF

Installation of $PACKAGE
System: $SYSTEM

This script installs SageMath and $PACKAGE.  It asks before
installing anything; answer y (yes) or n (no) and press Enter.
EOF

if [ -n "${PYTHONPATH:-}${PYTHONHOME:-}" ]; then
    printf '\nWarning: the variable PYTHONPATH or PYTHONHOME is set in your shell.  It can\n'
    printf 'make Sage load the wrong Python packages; if something fails, unset it.\n'
fi

# Free space where conda goes (about 2 GB to download, 5 GB installed).
check_space() {
    local dir=$1 kb
    while [ ! -d "$dir" ]; do dir=$(dirname "$dir"); done
    kb=$(df -Pk "$dir" | awk 'NR == 2 { print $4 }')
    if [ -n "$kb" ] && [ "$kb" -lt 8000000 ]; then
        printf '\nOnly %s GB of disk space is free in %s; SageMath needs about 6 GB.\n' \
            "$((kb / 1000000))" "$(show_path "$dir")"
        ask "Continue anyway?" || exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. macOS: Apple's Command Line Tools

if [ "$OS" = Darwin ] && ! xcode-select -p >/dev/null 2>&1; then
    cat <<EOF

$PACKAGE compiles part of itself on your Mac, which needs Apple's
Command Line Tools (about 1 GB, free from Apple).  They are not installed.
EOF
    if [ "$YES" = 0 ] && ask "Install them now?"; then
        xcode-select --install || true
        cat <<EOF

A window from Apple has opened: follow it to install the Command Line Tools
(this can take 5-15 minutes).  When it has finished, run this script again:

    bash install.sh

It will continue where it stopped.
EOF
        exit 0
    fi
    cat <<EOF

The Command Line Tools are required.  To install them yourself, run
    xcode-select --install
then run this script again.
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. conda

# The first conda installation found: --conda, an active conda, the usual
# folders.
find_conda() {
    local dir
    if [ -n "$CONDA_DIR" ]; then
        [ -x "$CONDA_DIR/bin/conda" ] && return 0
        [ -e "$CONDA_DIR" ] && die "$CONDA_DIR exists but is not a conda installation (no bin/conda in it)."
        return 1
    fi
    for dir in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" \
               "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
               "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3"; do
        if [ -n "$dir" ] && [ -x "$dir/bin/conda" ]; then
            CONDA_DIR=$dir
            return 0
        fi
    done
    CONDA_DIR="$HOME/miniforge3"
    return 1
}

if find_conda; then
    say "Using the conda installation in $(show_path "$CONDA_DIR")"
else
    cat <<EOF

SageMath is installed with conda, from the package collection conda-forge.
No conda installation was found, so the script can install Miniforge (a
minimal conda for conda-forge; free, BSD licence) in $(show_path "$CONDA_DIR").
It does not change your shell settings: conda is only switched on when you
type the command that the end of this script shows.
EOF
    ask "Install Miniforge in $(show_path "$CONDA_DIR")?" || {
        printf '\nNothing was installed.  The installation guide explains how to\n'
        printf 'install SageMath yourself: https://github.com/simonjwood/vertexalgebras#installation\n'
        exit 1
    }
    command -v curl >/dev/null 2>&1 \
        || die "downloading Miniforge needs the program curl.  On Ubuntu (also in WSL) install it with:  sudo apt install curl"
    check_space "$CONDA_DIR"
    TMPDIR_INSTALL=$(mktemp -d)
    INSTALLER="$TMPDIR_INSTALL/Miniforge3-$OS-$ARCH.sh"
    run curl -fL --progress-bar -o "$INSTALLER" "$MINIFORGE_URL/Miniforge3-$OS-$ARCH.sh"
    # -b: no questions, and no changes to the shell settings.
    run bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -rf "$TMPDIR_INSTALL"
fi

CONDA="$CONDA_DIR/bin/conda"
# Every package from conda-forge, also with Anaconda or Miniconda (whose
# default channels have no SageMath).
CHANNELS=(--override-channels -c conda-forge)

# ---------------------------------------------------------------------------
# 3. The Sage environment

# sage_version DIR: the Sage version in the environment DIR, or nothing.
sage_version() {
    "$1/bin/python" -c 'import sage.version; print(sage.version.version)' 2>/dev/null || true
}

# sage_ok VERSION: 0 when VERSION (10.10, 10.11.beta2, 11.0, ...) is at
# least MIN_SAGE.
sage_ok() {
    local major minor rest
    major=${1%%.*}
    rest=${1#*.}
    minor=${rest%%[!0-9]*}
    case "$major" in ''|*[!0-9]*) return 1 ;; esac
    [ -n "$minor" ] || minor=0
    [ "$major" -gt "$MIN_SAGE_MAJOR" ] && return 0
    [ "$major" -eq "$MIN_SAGE_MAJOR" ] && [ "$minor" -ge "$MIN_SAGE_MINOR" ]
}

# The environment to use: ENV_NAME if it is good or does not exist yet;
# otherwise (it exists with an old Sage or without Sage) ALT_ENV_NAME, so
# that an environment of someone else's is never changed.
ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"
CREATE_ENV=0
for attempt in 1 2; do
    ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"
    if [ ! -d "$ENV_DIR" ]; then
        CREATE_ENV=1
        break
    fi
    SAGE_VERSION=$(sage_version "$ENV_DIR")
    if [ -n "$SAGE_VERSION" ] && sage_ok "$SAGE_VERSION"; then
        say "Using the conda environment \"$ENV_NAME\", with SageMath $SAGE_VERSION"
        break
    fi
    if [ -n "$SAGE_VERSION" ]; then
        printf '\nThe conda environment "%s" has SageMath %s;\n' "$ENV_NAME" "$SAGE_VERSION"
        printf '%s needs SageMath %s or later.\n' "$PACKAGE" "$MIN_SAGE"
    else
        printf '\nThe conda environment "%s" exists, but SageMath is not installed in it.\n' "$ENV_NAME"
    fi
    if [ "$attempt" = 1 ] && [ "$ENV_NAME" != "$ALT_ENV_NAME" ]; then
        printf 'This script does not change it.  It can create a new environment\n'
        printf '"%s" instead, and leave "%s" as it is.\n' "$ALT_ENV_NAME" "$ENV_NAME"
        ask "Use the environment \"$ALT_ENV_NAME\" instead?" || {
            printf '\nNothing was changed.  To install into "%s", remove it first (see\n' "$ENV_NAME"
            printf '"Uninstalling" in the installation guide) or choose another name\n'
            printf 'with  bash install.sh --env NAME\n'
            exit 1
        }
        ENV_NAME=$ALT_ENV_NAME
    else
        die "choose another environment name with  bash install.sh --env NAME"
    fi
done

if [ "$CREATE_ENV" = 1 ]; then
    cat <<EOF

The script can now create the conda environment "$ENV_NAME" with SageMath,
Python, Jupyter and a C++ compiler, from conda-forge.  This downloads about
2 GB and takes 5-15 minutes; the environment needs about 5 GB of disk
space, in $(show_path "$ENV_DIR").
EOF
    ask "Create the environment \"$ENV_NAME\"?" || {
        printf '\nThe environment was not created; nothing else was changed.\n'
        exit 1
    }
    check_space "$CONDA_DIR"
    # A failed or interrupted creation leaves a half-made environment, which
    # a second run would take for someone else's: remove it (it did not
    # exist before).
    trap 'rm -rf "$ENV_DIR"; exit 130' INT TERM
    if ! run "$CONDA" create -y -n "$ENV_NAME" "${CHANNELS[@]}" "sage>=$MIN_SAGE" cxx-compiler pkg-config; then
        rm -rf "$ENV_DIR"
        die "conda could not create the environment (see above); nothing was kept."
    fi
    trap - INT TERM
    SAGE_VERSION=$(sage_version "$ENV_DIR")
    [ -n "$SAGE_VERSION" ] || die "SageMath does not start in the new environment \"$ENV_NAME\"."
    say "SageMath $SAGE_VERSION is installed"
else
    # An existing environment: the compiler may be missing.
    MISSING=
    for pkg in cxx-compiler pkg-config; do
        [ -d "$ENV_DIR/conda-meta" ] && ls "$ENV_DIR/conda-meta/$pkg"-[0-9]*.json >/dev/null 2>&1 || MISSING="$MISSING $pkg"
    done
    if [ -n "$MISSING" ]; then
        printf '\n%s compiles C++ code when it is installed, which needs\n' "$PACKAGE"
        printf 'these packages in the environment "%s":%s.\n' "$ENV_NAME" "$MISSING"
        ask "Install them (from conda-forge, about 200 MB)?" || {
            printf '\nNothing was changed.\n'
            exit 1
        }
        # shellcheck disable=SC2086
        run "$CONDA" install -y -n "$ENV_NAME" "${CHANNELS[@]}" $MISSING
    fi
fi

# ---------------------------------------------------------------------------
# 4. The package

# Switch the environment on (this sets the compiler variables too).  The
# activation scripts are not written for set -u.
set +eu
# shellcheck disable=SC1091
source "$CONDA_DIR/bin/activate" "$ENV_NAME"
STATUS=$?
set -eu
[ "$STATUS" = 0 ] && [ -n "${CONDA_PREFIX:-}" ] \
    && [ "$(cd "$CONDA_PREFIX" && pwd -P)" = "$(cd "$ENV_DIR" && pwd -P)" ] \
    || die "could not switch on the environment \"$ENV_NAME\"."

OLD_VERSION=$(python -c 'from importlib.metadata import version; print(version("sage-vertex-algebras"))' 2>/dev/null || true)
if [ -n "$OLD_VERSION" ]; then
    say "Upgrading $PACKAGE (installed: $OLD_VERSION); pip compiles it if there is a new version (about a minute)"
else
    say "Installing $PACKAGE; pip compiles its C++ part (about a minute)"
fi
# --no-cache-dir: never reuse a build from pip's cache, which may have been
# compiled against the libraries of another environment.
run python -m pip install --no-cache-dir --upgrade "$PIP_SPEC"

VERSION=$(python -c 'import vertex_algebras, vertex_algebras._universal_engine; print(vertex_algebras.__version__)') \
    || die "the package was installed, but importing it failed (above)."
say "$PACKAGE $VERSION works"

# On macOS, FLINT and GMP must come from the environment, not Homebrew.
if [ "$OS" = Darwin ]; then
    ENGINE=$(python -c 'import vertex_algebras._universal_engine as m; print(m.__file__)')
    if otool -L "$ENGINE" | grep -E 'lib(flint|gmp)' | grep -qE '/opt/homebrew|/usr/local'; then
        printf '\nWarning: the package was compiled against FLINT or GMP from Homebrew, not\n'
        printf 'from the conda environment; it may crash.  Please open an issue at\n    %s\n' "$ISSUES"
        printf 'with the output of:  otool -L %s\n' "$ENGINE"
    fi
fi

# ---------------------------------------------------------------------------
# 5. The example notebooks

NOTEBOOKS_READY=0
if [ "$COPY_NOTEBOOKS" = 1 ]; then
    if ! python -c 'import vertex_algebras.examples' 2>/dev/null; then
        printf '\nThis version of %s cannot copy its example notebooks (that\n' "$PACKAGE"
        printf 'came with version 0.1.0a5); they are in the folder examples of\n'
        printf 'https://github.com/simonjwood/vertexalgebras\n'
    elif [ -e "$NOTEBOOKS" ]; then
        cat <<EOF

The folder $(show_path "$NOTEBOOKS") exists already; it is not changed.  To get the
notebooks of version $VERSION, copy them into a new folder, for example:

    python -m vertex_algebras.examples ~/vertex-algebra-notebooks-$VERSION
EOF
        NOTEBOOKS_READY=1
    else
        cat <<EOF

The package includes 16 example notebooks and its test suite.  The script
can copy them to the folder $(show_path "$NOTEBOOKS"), where you can run
and change them (another folder: bash install.sh --notebooks FOLDER).
EOF
        if ask "Copy the example notebooks to $(show_path "$NOTEBOOKS")?"; then
            run python -m vertex_algebras.examples "$NOTEBOOKS"
            NOTEBOOKS_READY=1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Done

ACTIVATE="source $(show_path "$CONDA_DIR")/bin/activate $ENV_NAME"
cat <<EOF

==> Done: $PACKAGE $VERSION is installed, with SageMath $SAGE_VERSION,
    in the conda environment "$ENV_NAME".

To use it, open a new terminal and switch the environment on:

    $ACTIVATE

EOF
if [ "$NOTEBOOKS_READY" = 1 ]; then
    cat <<EOF
Then start Jupyter in the notebook folder:

    cd $(show_path "$NOTEBOOKS")
    sage -n jupyter

EOF
    if [ "$WSL" = 1 ]; then
        cat <<EOF
If no browser window opens, copy the address that starts with
http://localhost:8888/ (select it, then Ctrl+Shift+C) into a browser in
Windows.

EOF
    fi
    echo "Open 01_getting_started.ipynb first."
else
    echo "Then type  sage  to start Sage, and  quit  to leave it."
fi
cat <<EOF

To upgrade $PACKAGE later, run  bash install.sh  again.
EOF
