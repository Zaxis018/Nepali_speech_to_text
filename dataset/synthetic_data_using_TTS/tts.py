import os
import edge_tts
import asyncio
import sys
from tqdm import tqdm

async def text_to_speech(text, output_path):
    communicate = edge_tts.Communicate(text, voice="ne-NP-SagarNeural")
    await communicate.save(output_path)

async def convert_texts_to_speech(input_folder, output_folder):
    text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file_name in tqdm(text_files, desc="Converting files", unit="file"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace('.txt', '.mp3'))
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
            await text_to_speech(text, output_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python tts.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Make output folder if not exists already
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Run the conversion
    asyncio.run(convert_texts_to_speech(input_folder, output_folder))
    print(f'All text files have been converted to audio and saved in the "{output_folder}" folder.')

if __name__ == "__main__":
    main()

