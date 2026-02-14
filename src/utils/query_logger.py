from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid


class QueryLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any], *, request_id: Optional[str] = None) -> str:
        rid = request_id or str(uuid.uuid4())

        payload = {
            "request_id": rid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return rid
