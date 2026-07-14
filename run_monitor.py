import os

from market_oracle.monitor import monitor_loop


if __name__ == "__main__":
    interval_hours = float(os.getenv("MARKETSCOPE_SCAN_INTERVAL_HOURS", "6"))
    poll_seconds = int(os.getenv("MARKETSCOPE_MONITOR_POLL_SECONDS", "60"))
    monitor_loop(interval_hours=interval_hours, poll_seconds=poll_seconds)
