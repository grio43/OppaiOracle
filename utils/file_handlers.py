import logging
import logging.handlers
import gzip
import os
from typing import Optional


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    A lightweight RotatingFileHandler that optionally gzips rolled files.
    Pure stdlib; no project deps.
    """

    def __init__(self, filename, mode: str = 'a', maxBytes: int = 0, backupCount: int = 0,
                 encoding: Optional[str] = None, delay: bool = False, compress: bool = True) -> None:
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        # Whether to gzip rotated logs. When False, behaves like the standard
        # RotatingFileHandler.
        self.compress = compress

    def doRollover(self) -> None:
        """Rotate the log file and optionally compress the previous file."""
        # Close the current stream if open so we can safely move the file.
        if self.stream:
            self.stream.close()
            self.stream = None

        # Rotate existing files up to backupCount
        if self.backupCount > 0:
            # Move older backups up by one index (n -> n+1)
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}"
                dfn = f"{self.baseFilename}.{i+1}"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        try:
                            os.remove(dfn)
                        except Exception:
                            pass
                    try:
                        os.rename(sfn, dfn)
                    except Exception:
                        pass

            # Move the current log to index 1
            dfn = f"{self.baseFilename}.1"
            if os.path.exists(dfn):
                try:
                    os.remove(dfn)
                except Exception:
                    pass
            try:
                os.rename(self.baseFilename, dfn)
            except Exception:
                # If we can't rotate, reopen stream and bail
                if not self.delay:
                    self.stream = self._open()
                return

            # Compress the most recent rollover if requested
            if self.compress and os.path.exists(dfn):
                gz_path = f"{dfn}.gz"
                try:
                    with open(dfn, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                        f_out.writelines(f_in)
                    os.remove(dfn)
                except Exception:
                    # Best-effort compression; keep original if gzip fails
                    pass

        # Reopen the stream for the new log file if we're not delaying.
        if not self.delay:
            self.stream = self._open()

