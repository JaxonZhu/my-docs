# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
PORT          ?= 8000
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help check serve Makefile

# Remove stale pages, re-read every source, and fail on warnings.
check:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -E -a -n -W --keep-going $(SPHINXOPTS) $(O)
	@$(PYTHON) ../scripts/check_docs.py "$(BUILDDIR)/html"

# Rebuild first, then serve locally. Stop with Ctrl+C.
serve: check
	@$(PYTHON) -m http.server $(PORT) --bind 127.0.0.1 --directory "$(BUILDDIR)/html"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
