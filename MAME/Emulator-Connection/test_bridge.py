"""Proves Python can drive MAME inputs

HOW TO RUN:  python test_bridge.py   (needs the galaga88 ROM in MAME's roms folder; see MAME path below)

WHAT THIS DOES
  1. Launches MAME (the arcade emulator) with bridge.lua loaded inside it.
  2. Connects to MAME over a local TCP socket (MAME is the server on port 5000, we are the client).
  3. Every game frame, MAME sends us a line of text; we answer with ONE BYTE saying which
     buttons to hold next frame. MAME waits for that byte, so game and script stay in lockstep.
  4. We play a fixed script: insert coin, press start, move right, then move left + fire.
  5. MAME also reports what its emulated input ports actually registered. If that equals what we
     sent, the inputs reached the emulated arcade hardware (not just a keyboard stand-in): PASS.

Button bitmask (one bit per button; must match the order of MAP in bridge.lua):
  COIN=1, START=2, LEFT=4, RIGHT=8, FIRE=16. Combine with | (e.g. LEFT | FIRE = 20). 0 = nothing held.
"""
import socket, subprocess, time

MAME = r"C:\Users\ellis\OneDrive\Desktop\MAME" # CHANGE THIS TO YOUR OWN MAME FOLDER PATH
COIN, START, LEFT, RIGHT, FIRE = 1, 2, 4, 8, 16

# The "player":
SCRIPT = [(600, COIN), (610, 0), (700, START), (710, 0), (1100, RIGHT), (1300, LEFT | FIRE), (1450, 0)]

# Start MAME. Flags: -window (not fullscreen), -nothrottle (run as fast as the PC allows instead of
# real-time 60 fps), -sound none, -skip_gameinfo (skip the info screen),
# -autoboot_script (load our Lua bridge inside MAME as soon as it starts).
mame = subprocess.Popen([MAME + r"\mame.exe", "galaga88", "-window", "-nothrottle", "-sound", "none", "-skip_gameinfo",
                         "-autoboot_script", r"C:\Users\ellis\CSC410\bridge.lua"], cwd=MAME)

# MAME listens on :5000 (emu.file socket is a server); connect as client.
# MAME takes a moment to start up, so retry for up to ~20 s until its socket is open.
for _ in range(100):
    try: conn = socket.create_connection(("127.0.0.1", 5000)); break
    except OSError: time.sleep(0.2)
f = conn.makefile("r")  # lets us read MAME's messages one line at a time

# sent: the bitmask we sent last frame. seen: set of (sent, what MAME read back) pairs.
# prev_hash / changes: counts frames where game memory changed (shows the game is actually running).
sent, prev_hash, changes, seen = 0, None, 0, set()
for line in f:  # blocks until MAME finishes a frame and sends us its status line
    # Line format: "<frame number> <memory checksum> <button bitmask MAME's ports currently report>"
    frame, h, read = map(int, line.split())
    # `read` is what MAME's emulated input port reports for the mask we sent last frame
    if sent: seen.add((sent, read))
    if prev_hash is not None and h != prev_hash: changes += 1
    prev_hash = h
    # Decide what to hold next: the latest SCRIPT entry whose start frame has passed.
    # (An RL agent would replace this line with its policy choosing an action from the observation.)
    sent = max([m for fr, m in SCRIPT if fr <= frame], default=0)
    conn.sendall(bytes([sent]))  # one byte reply; MAME is waiting on this before running the next frame
    if frame >= 1600: break
mame.terminate()

# Verdict: every pair must match (what we sent == what MAME's ports reported), and all four
# distinct inputs we scripted must have been exercised.
print("sent->port-read pairs (should match):", sorted(seen))
ok = all(s == r for s, r in seen) and {COIN, START, RIGHT, LEFT | FIRE} <= {s for s, _ in seen}
print("RAM changed on", changes, "frames")
print("PASS" if ok else "FAIL")
