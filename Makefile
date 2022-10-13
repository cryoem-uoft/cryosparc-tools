# Mock target for cryosparc.core whose extension we don't know
TARGET=cryosparc_tools.egg-info/PKG-INFO

all: $(TARGET)

$(TARGET): src/*.c src/*.h cryosparc/*.pyx cryosparc/*.pxd setup.py pyproject.toml
	python -m setup develop

clean:
	rm -f cryosparc/*.so
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

.PHONY: clean all
