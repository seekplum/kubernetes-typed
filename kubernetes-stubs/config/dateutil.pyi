# Code generated by `stubgen`. DO NOT EDIT.
import datetime
from typing import Any

class TimezoneInfo(datetime.tzinfo):
    def __init__(self, h: Any, m: Any) -> None: ...
    def utcoffset(self, dt: Any): ...
    def tzname(self, dt: Any): ...
    def dst(self, dt: Any): ...

UTC: Any
MICROSEC_PER_SEC: int

def parse_rfc3339(s: Any): ...
def format_rfc3339(date_time: Any): ...