import sounddevice as sd
import numpy as np
from openwakeword.model import Model


class WakeWordDetector:

    def __init__(self):
        self.model = Model(
            wakeword_models=["hey_jarvis"]
        )

        self.sample_rate = 16000
        self.block_size = 1280

    def wait_for_wake_word(self):

        print("🎤 Waiting for 'Hey Jarvis'...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_size
        ) as stream:

            while True:

                audio, _ = stream.read(
                    self.block_size
                )

                audio = np.asarray(audio).flatten()

                prediction = self.model.predict(audio)

                for name, score in prediction.items():

                    if score > 0.5:

                        print("🟢 Wake word detected!")

                        return