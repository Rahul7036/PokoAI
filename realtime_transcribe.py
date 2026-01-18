import sounddevice as sd
import numpy as np
from google.cloud import speech
import queue

# ----------------------------
# Configuration
# ----------------------------
RATE = 16000
CHUNK = 1024  # Number of frames per buffer

q = queue.Queue()

# ----------------------------
# Callback for sounddevice stream
# ----------------------------
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    # Convert to bytes and push to queue
    q.put(indata.copy())

# ----------------------------
# Main function
# ----------------------------
def main():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    def audio_generator():
        while True:
            data = q.get()
            if data is None:
                return
            # Convert float32 array to int16 bytes
            audio_chunk = (data * 32767).astype(np.int16).tobytes()
            yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)

    # Open microphone stream
    with sd.InputStream(samplerate=RATE, channels=1, callback=callback, blocksize=CHUNK):
        print("Start speaking... Press Ctrl+C to stop.")
        requests = audio_generator()
        responses = client.streaming_recognize(streaming_config, requests)

        try:
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        print("Final:", transcript)
                    else:
                        print("Interim:", transcript, end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")
            q.put(None)

if __name__ == "__main__":
    main()



