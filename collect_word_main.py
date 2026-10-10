

from tqdm import tqdm
import os
import shutil


from globalVars import folder1, folder2
from pronounce_words_main import find_timestamps_file


def find_wav_by_word(folder_path, word):
    matching_files = [f for f in os.listdir(folder_path) 
                      if word.lower() in f.lower() and f.endswith(".wav")]
    return matching_files


def collect_word_main():
    text = 'decodewordspseudo'
    
    words = ['ποίημα', 'υπεύθυνος', 'βοοειδές', 'εκσυγχρονισμένος']
    subs = [subno for subno in range(14302, 14438)]

    for word in words:
        # Create word's destination folder
        dest_folder = os.path.join(folder1, folder2, 'clips', f"{word}_clips")
        os.makedirs(dest_folder, exist_ok=True)
        
        for sub_id in tqdm(subs, desc="Processing subjects"):

            sub_path = os.path.join(folder1, folder2, f"subject_{sub_id}")
            if not os.path.exists(sub_path):
                continue
            
            print(sub_id)
            
            timestamps_path = find_timestamps_file(sub_path, sub_id, text)
            word_clips_dir = os.path.join(os.path.dirname(timestamps_path), f"{text}_word_clips_{sub_id}")
            result = find_wav_by_word(word_clips_dir, word)
            
            if result:  # Check if file was found
                    source_file = word_clips_dir + '/' + result[0]  # Get the first match
                    dest_file_name = os.path.join(dest_folder, f"{sub_id}_{word}.wav")  # Use os.path.join
                    
                    shutil.copy(source_file, dest_file_name)
                    print(f"Copied to: {dest_file_name}")  # Print dest_file_name, not dest_file
            else:
                print(f"No WAV file with '{word}' found for {sub_id}")
                

collect_word_main()