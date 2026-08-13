import io
import sounddevice as sd
import soundfile as sf
from openai import OpenAI


client = OpenAI()


def speak(text):

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=text,
        response_format="wav"
    ) as response:

        audio_buffer = io.BytesIO()

        for chunk in response.iter_bytes(chunk_size=4096):
            audio_buffer.write(chunk)

            audio_buffer.seek(0)

            try:
                data, sample_rate = sf.read(
                    audio_buffer,
                    dtype="float32"
                )

                sd.play(data, sample_rate)
                sd.wait()

            except Exception:
                pass

            audio_buffer.seek(0)
            audio_buffer.truncate(0)