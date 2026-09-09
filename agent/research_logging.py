"""Console verbosity; persisted research events remain complete."""
import json


class ResearchLogger:
    def __init__(self, level="basic"):
        if level not in {"basic", "full"}:
            raise ValueError("log_level must be basic or full")
        self.level = level

    def show(self, label, payload, *, detailed=False):
        if detailed and self.level != "full":
            return
        body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        print(f"\n{label}\n{body}", flush=True)
