"""
Whistle-to-serve.

The same idea as the whistle spectrogram from class, done live: every
~46 ms block of microphone audio gets an FFT (one column of a
spectrogram). A whistle is a nearly pure tone, so its column has one very
tall, narrow peak between 600 and 4000 Hz while everything else stays low.
We measure "tonality" = peak power / median power in that band. When enough
consecutive blocks are tonal (about 0.3 s of steady whistle), that's a
serve. Talking, clapping and fan noise spread their energy across many
frequencies, so their tonality stays low.

Run this file directly to tune the thresholds against your mic and room:
    python whistle.py
"""

import queue
import threading

import numpy as np

import config


def analyze_block(block, sample_rate=config.MIC_SAMPLE_RATE):
    """One spectrogram column -> (is_whistle_like, peak_hz, tonality, rms)."""
    block = np.asarray(block, dtype=np.float64)
    rms = float(np.sqrt(np.mean(block ** 2)))
    spectrum = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
    freqs = np.fft.rfftfreq(len(block), 1 / sample_rate)
    lo, hi = config.WHISTLE_BAND_HZ
    band = (freqs >= lo) & (freqs <= hi)
    band_power = spectrum[band]
    peak_i = int(np.argmax(band_power))
    tonality = float(band_power[peak_i] / (np.median(band_power) + 1e-12))
    peak_hz = float(freqs[band][peak_i])
    whistle_like = rms >= config.WHISTLE_MIN_RMS and tonality >= config.WHISTLE_TONALITY
    return whistle_like, peak_hz, tonality, rms


class WhistleDetector:
    """Counts consecutive whistle-like blocks; fires once per whistle."""

    def __init__(self):
        self._run = 0
        self._fired = False
        self.last = (False, 0.0, 0.0, 0.0)

    def feed(self, block):
        self.last = analyze_block(block)
        if self.last[0]:
            self._run += 1
        else:
            self._run = 0
            self._fired = False
        if self._run >= config.WHISTLE_MIN_BLOCKS and not self._fired:
            self._fired = True
            return True
        return False


class WhistleListener:
    """Runs the mic in the background. poll() returns True once per whistle."""

    def __init__(self, device=None):
        import sounddevice as sd

        self._detector = WhistleDetector()
        self._events = queue.Queue()
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=config.MIC_SAMPLE_RATE, blocksize=config.MIC_BLOCK,
            channels=1, dtype="float32", device=device, callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            if self._detector.feed(indata[:, 0]):
                self._events.put(True)

    @property
    def level(self):
        """(whistle_like, peak_hz, tonality, rms) of the latest block, for the HUD."""
        return self._detector.last

    def poll(self):
        fired = False
        while not self._events.empty():
            self._events.get_nowait()
            fired = True
        return fired

    def close(self):
        self._stream.stop()
        self._stream.close()


if __name__ == "__main__":
    import time
    print("Whistle at the mic. Ctrl+C to stop.")
    listener = WhistleListener()
    try:
        while True:
            like, hz, ton, rms = listener.level
            bar = "#" * min(40, int(ton / 5))
            print(f"\r{'WHISTLE' if like else '       '} {hz:6.0f} Hz  tonality {ton:6.1f} "
                  f"rms {rms:.3f} {bar:<40}", end="", flush=True)
            if listener.poll():
                print("\n>>> SERVE!")
            time.sleep(0.05)
    except KeyboardInterrupt:
        listener.close()
