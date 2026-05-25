import socket
import numpy as np

# ============================================================
# UDP SETTINGS
# ============================================================

HOST_IP   = "0.0.0.0"
HOST_PORT = 5005

# ============================================================
# CREATE UDP SOCKET
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind socket
sock.bind((HOST_IP, HOST_PORT))

print(f"Listening on UDP port {HOST_PORT}...\n")

# ============================================================
# RECEIVE LOOP
# ============================================================

while True:

    # Receive packet
    data, addr = sock.recvfrom(65536)

    print(f"Packet received from {addr}")

    # Convert bytes back to complex64 IQ
    iq = np.frombuffer(data, dtype=np.complex64)

    print("Number of IQ samples:", len(iq))

    # Print first 10 samples
    print(iq[:10])

    print("-" * 60)