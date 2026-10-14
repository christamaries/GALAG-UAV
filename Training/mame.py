import socket
import subprocess
import time
from pathlib import Path


# Configure these paths for your system.
MAME = "/opt/homebrew/bin/mame"
ROMS = "/Users/jacobigielski/mame/roms"
BRIDGE = Path(__file__).resolve().parent.parent / "MAME/Emulator-Connection/bridge.lua"

# These bits match MAP in bridge.lua.
COIN, START, LEFT, RIGHT, FIRE = 1, 2, 4, 8, 16
ACTIONS = (
    0,             # Do nothing
    LEFT,
    RIGHT,
    FIRE,
    LEFT | FIRE,
    RIGHT | FIRE,
)
FRAMES_PER_STEP = 4


class MameBridge:
    def __init__(self):
        self.process = None
        self.conn = None
        self.reader = None
        self.frame = 0
        self.input_readback = 0

    def start(self):
        self.close()

        try:
            self.process = subprocess.Popen([
                MAME, "galaga88", "-rompath", ROMS, "-window", "-nothrottle",
                "-sound", "none", "-skip_gameinfo", "-autoboot_script", str(BRIDGE),
            ])

            # MAME's Lua bridge listens on this port; Python connects to it.
            for _ in range(100):
                try:
                    self.conn = socket.create_connection(("127.0.0.1", 57321))
                    break
                except OSError:
                    if self.process.poll() is not None:
                        raise RuntimeError("MAME exited before the bridge connected")
                    time.sleep(0.2)
            if self.conn is None:
                raise TimeoutError("MAME did not open the bridge socket on port 57321")

            self.reader = self.conn.makefile("r")
            self._read_frame()

            # The existing bridge test inserts a coin at frame 600 and starts at 700.
            while self.frame < 1100:
                if 600 <= self.frame < 610:
                    mask = COIN
                elif 700 <= self.frame < 710:
                    mask = START
                else:
                    mask = 0
                self.advance(mask)
        except Exception:
            self.close()
            raise

    def advance(self, mask, frames=1):
        if self.conn is None:
            raise RuntimeError("Call reset() before step()")
        for _ in range(frames):
            self.conn.sendall(bytes([mask]))
            self._read_frame()

    def _read_frame(self):
        line = self.reader.readline()
        if not line:
            raise RuntimeError("MAME bridge disconnected")
        self.frame, _, self.input_readback = map(int, line.split())

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
