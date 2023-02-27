# Actual target is a .so file with a dynamically-determined name, but this is
# close enough
TARGET=cryosparc/core.c
PYTHON=python3.9
PYTHON_VERSION=3.9.16

all: $(TARGET)

# -----------------------------------------------------------------------------
#    Primary build target
# -----------------------------------------------------------------------------

$(TARGET): cryosparc/include/cryosparc-tools/*.h cryosparc/dataset.c cryosparc/*.pyx cryosparc/*.pxd setup.py pyproject.toml
	python -m setup build_ext -i

# -----------------------------------------------------------------------------
#    Vercel deployment-related targets
# -----------------------------------------------------------------------------

Python-$(PYTHON_VERSION).tgz:
	curl -L https://www.python.org/ftp/python/$(PYTHON_VERSION)/Python-$(PYTHON_VERSION).tgz -o Python-$(PYTHON_VERSION).tgz

Python-$(PYTHON_VERSION): Python-$(PYTHON_VERSION).tgz
	tar -xzf Python-${PYTHON_VERSION}.tgz

python: Python-$(PYTHON_VERSION)
	cd Python-$(PYTHON_VERSION) && ./configure && cd .. || cd ..
	make -C Python-$(PYTHON_VERSION)
	make -C Python-$(PYTHON_VERSION) altinstall

.venv/bin/python:
	$(PYTHON) -m venv .venv

.venv/bin/pip: .venv/bin/python
	.venv/bin/python -m pip install -U pip

.venv/bin/jupyter-book: .venv/bin/pip
	.venv/bin/pip install -e ".[build]"

.vercel/output/config.json:
	mkdir -p .vercel/output
	echo '{"version":3,"cache":[".venv/**","build/**","docs/_build/**","Python-*.tgz","Python-*/**"]}' > .vercel/output/config.json

verceldeps:
	yum update -y
	yum install bzip2-devel libffi-devel openssl-devel sqlite-devel -y

vercelinstall: verceldeps python .venv/bin/python
	echo "Install complete"

vercelbuild: .vercel/output/config.json .venv/bin/jupyter-book
	.venv/bin/jupyter-book build docs
	rm -rf .vercel/output/static && cp -R docs/_build/html .vercel/output/static

# -----------------------------------------------------------------------------
#    Cleanup
# -----------------------------------------------------------------------------

clean:
	rm -f $(TARGET)
	rm -f *.tgz *.tar.gz
	rm -f cryosparc/*.so
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf Python-$(PYTHON_VERSION)

.PHONY: clean all python verceldeps vercelinstall vercelbuild
