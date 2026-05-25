import adi
import numpy as np
import socket
import time

# ============================================================
# NETWORK SETTINGS
# ============================================================

HOST_IP   = "192.168.18.42"   # Your laptop IP
HOST_PORT = 5005

# ============================================================
# CREATE UDP SOCKET
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ============================================================
# CONNECT TO SDR
# ============================================================

sdr = adi.ad9364("local:")

# ============================================================
# SDR SETTINGS
# ============================================================

sdr.sample_rate = int(1e6)
sdr.rx_lo = int(433e6)
sdr.rx_rf_bandwidth = int(200e3)
sdr.rx_buffer_size = 512
sdr.rx_hardwaregain_chan0 = 0

print("SDR initialized...")
print("Sending IQ samples to host...\n")

# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        # Receive IQ samples from SDR
        iq = sdr.rx()

        # Convert complex IQ -> complex64
        iq = np.array(iq, dtype=np.complex64)

        # Convert to raw bytes
        iq_bytes = iq.tobytes()

        # Send UDP packet
        sock.sendto(iq_bytes, (HOST_IP, HOST_PORT))

        print(f"Sent {len(iq)} IQ samples")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping transmitter...")

finally:
    try:
        sdr.rx_destroy_buffer()
    except:
        pass

    sock.close()
    del sdr

    print("Resources released.")