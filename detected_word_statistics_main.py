from tqdm import tqdm
import os
import pandas as pd
from globalVars import folder1, folder2
from pronounce_words_main import find_timestamps_file


def find_detected_words_excel(folder_path, sub_id, text):
    """Find the detected words Excel file for a subject"""
    matching_files = [f for f in os.listdir(folder_path) 
                      if text.lower() in f.lower() and 'decodewordspseudo_detected_words_v3' in f.lower() and f.endswith(".xlsx")]
    return matching_files



def detected_word_statistics_main():
    text = 'decodewordspseudo'
    
    words = ['ποίημα', 'υπεύθυνος', 'βοοειδές', 'εκσυγχρονισμένος']
    
    subs = [subno for subno in range(14302, 14438)]
    
    for word in words:
        # Create word's statistics folder
        stats_folder = os.path.join(folder1, folder2, 'statistics', f"{word}_statistics")
        os.makedirs(stats_folder, exist_ok=True)
        
        # Lists to collect results
        results_data = {
            'subject_id': [],
            'detected_word': []
        }
        
        count_occurrences = {}
        
        for sub_id in tqdm(subs, desc=f"Processing subjects for '{word}'"):
            sub_path = os.path.join(folder1, folder2, f"subject_{sub_id}")
            if not os.path.exists(sub_path):
                continue
            
            try:
                timestamps_path = find_timestamps_file(sub_path, sub_id, text)
                detected_words_dir = os.path.dirname(timestamps_path)
                
                # Find the Excel file with detected words
                excel_files = find_detected_words_excel(detected_words_dir, sub_id, text)
                
                if not excel_files:
                    continue
                
                # Read the Excel file
                excel_path = os.path.join(detected_words_dir, excel_files[0])
                df = pd.read_excel(excel_path)
                
                # Find the row with the target word
                word_row = df[df['word'].str.lower() == word.lower()]
                
                if not word_row.empty:
                    detected = word_row.iloc[0]['matched_transcription']
                    
                    results_data['subject_id'].append(sub_id)
                    results_data['detected_word'].append(detected)
                    
                    count_occurrences[detected] = count_occurrences.get(detected, 0) + 1
                  
                
            except Exception as e:
                print(f"Error processing subject {sub_id}: {e}")
                continue
        
        # Create DataFrame and save to Excel
        if results_data['subject_id']:  # Only save if we have data
            results_df = pd.DataFrame(results_data)
            output_file = os.path.join(stats_folder, f"{word}_statistics.xlsx")
            results_df.to_excel(output_file, index=False)
            print(f"\nSaved statistics for '{word}' to: {output_file}")
            print(f"Total matches found: {len(results_df)}")
        else:
            print(f"\nNo data found for word '{word}'")


        if count_occurrences:
            result_occurencies = pd.DataFrame(list(count_occurrences.items()), columns=['Detected Word', 'Count'])
            output_file = os.path.join(stats_folder, f"{word}_occurencies_statistics.xlsx")
            result_occurencies.to_excel(output_file, index=False)
            print(f"Saved occurrence statistics to: {output_file}")
        
detected_word_statistics_main()