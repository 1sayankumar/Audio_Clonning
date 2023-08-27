##importing libraries 
import googletrans
import speech_recognition
import gtts
import playsound
# print(googletrans.LANGUAGES)

#translator = googletrans.Translator()
#translation = translator.translate("hello", dest = "hi")
#print(translation)
## Recognizing the voice and text using google translator
recognizer = speech_recognition.Recognizer()
with speech_recognition.AudioFile('/Users/MY PC/speech.wav') as source:
  Audio= recognizer.listen(source)
  text = recognizer.recognize_google(Audio,language="en")
  print(text)
  
translator = googletrans.Translator()
translation = translator.translate(text, dest = "hi")
print(translation.text)

## Here i used git and pip commands to clone the voice using tortoise library
!pip3 install -U scipy

!git clone https://github.com/jnordberg/tortoise-tts.git
%cd tortoise-tts
!pip3 install -r requirements.txt
!pip3 install transformers==4.19.0 einops==0.5.0 rotary_embedding_torch==0.1.5 unidecode==1.3.5
!python3 setup.py install

## importing pytorch and analysing the audio file and using text-to-speech api
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

## Here is the text which i have used to clone by using a lady's voice
text = "Hey my name is Natalie with another April English podcast so today is Monday and I am here in San Francisco and back in San Francisco last week we were in Texas we were at a teaching conference and the leader of the teaching conference was a plane I just want to say some nice things about person is a fantastic teacher is a great teacher training"
# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "fast"

## this section is basically used upload the original audio file.
CUSTOM_VOICE_NAME = "lady"

import os
from google.colab import files
custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
os.makedirs(custom_voice_folder)
for i, file_data in enumerate(files.upload().values()):
  with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
    f.write(file_data)

## this is the part which is used for cloning
voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=preset)
torchaudio.save(f'generated-{CUSTOM_VOICE_NAME}.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio(f'generated-{CUSTOM_VOICE_NAME}.wav')
