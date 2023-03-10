from vosk import Model, KaldiRecognizer
import pyaudio
from chat import get_response
model = Model(r"C:\Users\dell\Desktop\vosk-model-en-us-daanzu-20200905-lgraph\vosk-model-en-us-daanzu-20200905-lgraph")
recognizer = KaldiRecognizer(model, 16000)
def listen(audio):
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    while True:
        data = stream.read(8192)

        if recognizer.AcceptWaveform(data):
            text = recognizer.Result()
            print(f"' {text[14:-3]} '")
            return text
            break
