# Actual target is a .so file with a dynamically-determined name, but this is
# close enough
TARGET=cryosparc/core.c

all: $(TARGET)

$(TARGET): src/*.c src/*.h cryosparc/*.pyx cryosparc/*.pxd setup.py pyproject.toml
	python -m setup build_ext -i

clean:
	rm -f cryosparc/*.so
	rm -f cryosparc/*.c
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

.PHONY: clean all
