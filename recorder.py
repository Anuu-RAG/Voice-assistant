import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1
SILENCE_DURATION = 0.8
THRESHOLD = 0.01

BLOCK_SIZE = int(
    SAMPLE_RATE * BLOCK_DURATION
)


def record_until_silence():

    audio_chunks = []

    silence_time = 0
    started_speaking = False

    print("👂 Listening...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE
    ) as stream:

        while True:

            chunk, _ = stream.read(BLOCK_SIZE)

            volume = np.sqrt(
                np.mean(chunk ** 2)
            )

            if volume > THRESHOLD:

                started_speaking = True
                silence_time = 0

                audio_chunks.append(
                    chunk.copy()
                )

            else:

                if started_speaking:

                    silence_time += BLOCK_DURATION

                    audio_chunks.append(
                        chunk.copy()
                    )

                    if silence_time >= SILENCE_DURATION:
                        break

    audio = np.concatenate(audio_chunks)

    sf.write(
        "input.wav",
        audio,
        SAMPLE_RATE
    )

    print("✅ Recording finished")

    return "input.wav"