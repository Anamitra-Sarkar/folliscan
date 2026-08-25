"""Import-path shim so tests can import services.user_api.main."""

import sys
from pathlib import Path

services_dir = Path(__file__).resolve().parent.parent / "services" / "user-api"
sys.path.insert(0, str(services_dir))
