import sounddevice as sd
import soundfile as sf

samplerate = 44100
duration = 60
filename = './data/train_data2.wav'

print("start")
mydata = sd.rec(int(samplerate * duration),
                samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)
