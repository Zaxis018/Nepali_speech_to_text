import sys
import textwrap

from transformers import pipeline

def transcribe(pipe, audio_path):
    result = pipe(audio_path)
    return result['text']
    

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('Uses: python inference.py audio_file.mp3')
        sys.exit(0)
        
    
    audio_path  = sys.argv[1]
    
    pipe = pipeline("automatic-speech-recognition", model = 'amitpant7/Nepali-Automatic-Speech-Recognition')
    ts = transcribe(pipe, audio_path)
    ts = textwrap.fill(ts, width=50)
    print(ts)