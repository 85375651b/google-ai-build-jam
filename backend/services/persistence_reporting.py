"""Persistence success comes from committed results, never consultant prose."""
import re

CLAIM = re.compile(r"\b(?:saved|updated|recorded|committed|renamed|written|added|completed|all set|done)\b|\b(?:is now|name is now)\b", re.I)


class PendingReplyFilter:
    def __init__(self):
        self.buffer = ''

    def feed(self, value, final=False):
        self.buffer += value
        parts = re.split(r'(?<=[.!?])\s+|\n', self.buffer)
        self.buffer = '' if final else parts.pop()
        visible = '\n'.join(part for part in parts if part.strip() and not CLAIM.search(part))
        return visible + ('\n' if visible else '')

    def finish(self):
        return self.feed('', final=True)


def pending_reply(text):
    filter = PendingReplyFilter()
    return filter.feed(text, final=True) or 'Your response is ready. World Bible updates are awaiting validation.'
