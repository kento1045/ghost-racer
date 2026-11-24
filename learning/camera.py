"""One indexed FPV recording per flight, with the detector output from each control step."""
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

import cv2


class FlightCamera:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.db = None
        self.path = None
        self.pending_reset = None
        self.count = 0

    def capture(self, rgb, state, *, reset=False):
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 83]
        )
        if not ok:
            raise RuntimeError("カメラ画像を保存できませんでした")
        frame = (json.dumps(state, allow_nan=False), jpeg.tobytes())
        if reset:
            # SB3 auto-resets BEFORE its callback saves the previous terminal flight.
            self.pending_reset = frame
            return
        if self.db is None:
            with NamedTemporaryFile(dir=self.directory, suffix=".partial", delete=False) as temp:
                self.path = Path(temp.name)
            self.db = sqlite3.connect(self.path)
            self.db.execute("CREATE TABLE frames (i INTEGER PRIMARY KEY, state TEXT, jpeg BLOB)")
        if self.pending_reset is not None:
            self._append(self.pending_reset)
            self.pending_reset = None
        self._append(frame)

    def _append(self, frame):
        self.db.execute("INSERT INTO frames VALUES (?, ?, ?)", (self.count, *frame))
        self.count += 1

    def finish(self, trial):
        if self.db is None:
            return 0
        self.db.commit()
        self.db.close()
        self.db = None
        self.path.replace(self.directory / f"camera-{trial}.sqlite")
        count, self.count = self.count, 0
        self.path = None
        return count

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.path:
            self.path.unlink(missing_ok=True)


def read_recording(path, frame=None):
    if not path.is_file():
        return None
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        if frame is None:
            return [json.loads(s) for (s,) in db.execute("SELECT state FROM frames ORDER BY i")]
        row = db.execute("SELECT jpeg FROM frames WHERE i=?", (frame,)).fetchone()
        return row[0] if row else None
    finally:
        db.close()
