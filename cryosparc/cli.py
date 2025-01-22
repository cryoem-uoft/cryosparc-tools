import hashlib
import os
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime, timedelta, timezone
from getpass import getpass

from . import __version__
from .api import APIClient
from .auth import InstanceAuthSessions, get_default_auth_config_path
from .constants import API_SUFFIX
from .errors import APIError


def run(name: str = "cryosparc.tools"):
    parser = ArgumentParser(prog=name)
    parser.add_argument("-v", "--version", action="version", version=f"cryosparc-tools {__version__}")
    subparsers = parser.add_subparsers(help="command help")

    # Login command
    parser_login = subparsers.add_parser(
        "login",
        help="log in to CryoSPARC and store authentication token in the home directory for repeated script runs",
    )
    parser_login.add_argument(
        "--url",
        type=str,
        required=False,
        help="CryoSPARC web URL, e.g., http://localhost:39000",
    )
    parser_login.add_argument("--email", type=str, required=False, help="login email")
    parser_login.add_argument("--password", type=str, required=False, help="login password")
    parser_login.add_argument(
        "--expires",
        type=valid_expiration_time,
        help="token expiration date in format YYYY-MM-DD. Cannot be more than one year in the future. "
        "Defaults to 14 days from now.",
        required=False,
    )
    parser_login.set_defaults(func=login)

    args = parser.parse_args()
    args.func(args)


def login(args: Namespace):
    expires_in = 60 * 60 * 24 * 14
    if args.expires:
        expires_in = args.expires
    if not args.url:
        args.url = input("CryoSPARC URL: ")
    if not args.email:
        args.email = input("Email: ")
    if not args.password:
        args.password = getpass("Password: ")

    sessions = InstanceAuthSessions.load()

    expiration_date = datetime.now() + timedelta(seconds=expires_in)
    try:
        # TODO: Correct URL
        api = APIClient(f"{args.url}{API_SUFFIX}")
        token = api.login(
            grant_type="password",
            username=args.email,
            password=hashlib.sha256(args.password.encode()).hexdigest(),
            expires_in=expires_in,
        )
    except APIError as err:
        print(err, file=sys.stderr)
        print("Login failed!", file=sys.stderr)
        os._exit(1)

    sessions.insert(args.url, args.email, token, expiration_date.astimezone(timezone.utc))
    sessions.save()
    print(f"Success! Login token for {args.url} ({args.email}) saved to {get_default_auth_config_path()}")


def valid_expiration_time(date_str: str) -> float:
    """
    Returns the expiration time of a token if it is within the next year.
    """
    try:
        expires_at = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ArgumentTypeError(f"Not a valid date: '{date_str}'. Format should be YYYY-MM-DD.")
    min_expires_in = 1
    max_expires_in = 60 * 60 * 24 * 365
    now = datetime.now()
    expires_in = (expires_at - now).total_seconds()
    if not (min_expires_in <= expires_in <= max_expires_in):
        raise ArgumentTypeError(f"Not a valid expiration date: '{date_str}'. Must within the next year")
    return expires_in
