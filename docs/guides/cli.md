# Command Line Interface

cryosparc-tools includes a simple command line interface (CLI) for managing your CryoSPARC instance.

After [installation](/intro), run commands from your terminal in this format:

```bash
python -m cryosparc.tools <command> [options]
```

Currently, this only supports the `login` command.

Run with the `--help` flag to see available commands and options:

```bash
python -m cryosparc.tools --help
```

## Commands

### `login`

Log in to a CryoSPARC instance.

```bash
python -m cryosparc.tools login --url <URL>
```

Replace `<URL>` with the URL you use to access CryoSPARC from your web browser, e.g., `http://localhost:39000`. Enter your CryoSPARC email and password when prompted.

This saves a login token to a local file, used to authenticate scripts that use the `CryoSPARC` function without needing to include your credentials in the script:

```python
from cryosparc.tools import CryoSPARC

cs = CryoSPARC("http://localhost:39000")
project = cs.find_project("P1")
```

The token expires after 2 weeks by default. To renew it, run the login command again. Specify flag `--expires <DATE>`, with `<DATE>` in `YYYY-MM-DD` format, to set a longer expiration date. For example:

```bash
python -m cryosparc.tools login --url http://localhost:39000 --expires 2026-12-31
```

This sets the token to expire on December 31, 2026. Note that the expiration date cannot be more than 1 year in the future.

You may also specify one or both of `--email` and `--password` flags to provide your credentials directly.

```bash
python -m cryosparc.tools login --url http://localhost:39000 \
  --email "ali@example.com" \
  --password "mysecretpassword" \
  --expires 2026-12-31
```

#### Advanced Usage

You may log in to multiple CryoSPARC instances or with different user accounts by running the command multiple times with different URLs and email addresses. For example:

```bash
python -m cryosparc.tools login --url http://localhost:39000 --email ali@example.com
python -m cryosparc.tools login --url http://localhost:39000 --email saara@example.com
python -m cryosparc.tools login --url https://cryosparc.example.com --email suhail@example.com
```

In your Python script, specify an `email` argument to use a specific login when multiple logins are active:

```python
from cryosparc.tools import CryoSPARC

local_cs = CryoSPARC("http://localhost:39000", email="saara@example.com")
remote_cs = CryoSPARC("https://cryosparc.example.com", email="suhail@example.com")
```

If no email is specified, the first login created for the given URL will be used.
