import asyncio
import struct
import argparse
from collections import deque
from typing import Tuple

import numpy as np
from scipy import signal

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except Exception:
    RandomForestClassifier = None
    train_test_split = None
    classification_report = None
    joblib = None


"""
Async UDP IQ pipeline for inference/training.

This script provides:
- an async UDP receiver that parses a DI header + IQ samples
- preprocessing (IQ -> complex, DC removal, normalization)
- feature extraction (time-domain stats, FFT bands, spectrogram summary)
- async pipeline using asyncio.Queue to ensure non-blocking receive
- simple train/inference selection (1=train, 2=infer)

Usage examples:
 python ScriptedPipeline.py --mode 2 --host 0.0.0.0 --port 10001
 python ScriptedPipeline.py --mode 1 --host 0.0.0.0 --port 10001 --model out.pkl

Note: sklearn and joblib recommended for training; if not installed, training is disabled.
"""


def parse_di_udp_packet(data: bytes) -> Tuple[dict, np.ndarray]:
    """Parse a simple DI header then IQ int16 pairs.

    Header format (example): 4sI -> 4-byte tag, 4-byte payload length
    Followed by payload of int16 interleaved I,Q
    """
    if len(data) < 8:
        raise ValueError("Packet too small")
    tag, plen = struct.unpack_from("4sI", data, 0)
    header = {"tag": tag.decode(errors="ignore"), "plen": plen}
    raw = data[8:8 + plen]
    # interpret as int16
    iq = np.frombuffer(raw, dtype=np.int16)
    if iq.size == 0:
        raise ValueError("Empty IQ payload")
    if iq.size % 2 != 0:
        iq = iq[:-1]
    if iq.size == 0:
        raise ValueError("Incomplete IQ pair")
    iq = iq.reshape(-1, 2)
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    return header, complex_iq


def preprocess(iq: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Basic preprocessing: DC removal, lowpass optional, normalization."""
    # DC removal
    iq = iq - np.mean(iq)
    # simple bandpass via detrend + normalize
    iq = signal.detrend(iq)
    power = np.sqrt(np.mean(np.abs(iq) ** 2)) + 1e-12
    iq = iq / power
    return iq


def extract_features(iq: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Extract features suitable for classical ML.

    Features: real/imag mean/std, amplitude moments, FFT band energies, spectrogram summaries
    """
    real = np.real(iq)
    imag = np.imag(iq)
    amp = np.abs(iq)

    feats = []
    feats += [np.mean(real), np.std(real), np.mean(imag), np.std(imag)]
    feats += [np.mean(amp), np.std(amp), np.max(amp), np.min(amp)]

    # FFT band energies
    f, Pxx = signal.welch(iq, fs=fs, nperseg=min(256, len(iq)))
    Pxx = np.abs(Pxx)
    # split into 4 bands
    bands = np.array_split(Pxx, 4)
    for b in bands:
        feats.append(np.sum(b))

    # spectrogram summary (magnitude mean and variance)
    try:
        f2, t2, Sxx = signal.spectrogram(iq, fs=fs, nperseg=min(128, len(iq)))
        S = np.abs(Sxx)
        feats += [np.mean(S), np.std(S)]
    except Exception:
        feats += [0.0, 0.0]

    return np.array(feats, dtype=np.float32)


class AsyncUDPPipeline:
    def __init__(self, host: str, port: int, model_path: str = "model.pkl"):
        self.host = host
        self.port = port
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self.model_path = model_path
        self.model = None

    async def start_receiver(self):
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.queue), local_addr=(self.host, self.port)
        )
        print(f"Listening UDP {self.host}:{self.port}")

    async def worker(self, infer: bool = True):
        while True:
            data = await self.queue.get()
            try:
                header, iq = parse_di_udp_packet(data)
            except Exception as e:
                print("Parse error:", e)
                continue
            try:
                iq = preprocess(iq)
            except Exception as e:
                print("Preprocess error:", e)
                continue
            feats = extract_features(iq)
            if infer:
                if self.model is None:
                    # lazy load
                    try:
                        self.model = joblib.load(self.model_path)
                        print("Model loaded")
                    except Exception:
                        print("No model available for inference")
                        continue
                pred = self.model.predict(feats.reshape(1, -1))
                print("Pred:", pred)
            else:
                # In training mode, store features to buffer or disk; here we print shape
                print("Feature vector ready", feats.shape)


class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def datagram_received(self, data: bytes, addr):
        try:
            print(f"Packet received from {addr}, {len(data)} bytes")
        except Exception:
            pass
        try:
            # put_nowait to avoid blocking the socket reader
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            # drop packet if overwhelmed
            pass


async def run_pipeline(host: str, port: int, mode: int, model_path: str):
    pipeline = AsyncUDPPipeline(host, port, model_path)
    await pipeline.start_receiver()
    if mode == 2:
        await pipeline.worker(infer=True)
    else:
        await pipeline.worker(infer=False)


def train_model(X: np.ndarray, y: np.ndarray, out_path: str):
    if RandomForestClassifier is None:
        print("sklearn not available; cannot train")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, out_path)
    print("Model saved to", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=2, help="1=train, 2=infer")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--model", type=str, default="model.pkl")
    args = parser.parse_args()

    if args.mode == 1:
        # Placeholder: user should collect features and labels and call train_model
        print("Training mode selected. Implement data collection to call train_model().")
        return

    try:
        asyncio.run(run_pipeline(args.host, args.port, args.mode, args.model))
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
