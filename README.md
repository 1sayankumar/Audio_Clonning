# Audio_Cloning
Creating audio clones involves generating audio that sounds like a particular person or voice, often referred to as voice synthesis or voice cloning. This is often achieved through a process called voice adaptation, where the model is fine-tuned or adapted to a target speaker's voice using a small amount of their recorded speech data. 
# system_requirements
I have used spyder platform to translate the audio file into text the text and also used colab for audio cloning.
# Librarys 
Google translator, Speech_recognition, and google text to speech library, tortoise api to import text-to-speech command, pytorch
# process
i) Gather a substantial amount of audio data from the target speaker whose voice you want to clone.
ii) Align the collected audio with its corresponding transcripts. This will establish the relationship between spoken words and the actual text.
iii) Then I have train or fine-tune a hugging face model on our aligned dataset. This model converts input text into mel spectrograms that represent the speech characteristics.
iv) When I have a new text want to clone in the target voice, input it to the trained model to get the corresponding mel spectrogram, and then use the hugging face vocoder to clone the audio.
