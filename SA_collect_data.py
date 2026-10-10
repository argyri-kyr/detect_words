


import os
from tqdm import tqdm
import pandas as pd
import xlsxwriter


from globalVars import folder1, folder2


def collect_data_main():
    # 1. Create a new workbook and add a worksheet
    results_path = os.path.join(folder1, folder2, f"collected_data.xlsx")
    workbook = xlsxwriter.Workbook(results_path)
    worksheet = workbook.add_worksheet()
    
    # 2. Define text, expected words and subjects
    # 2. Write expected words in the first column of the worksheet
    text = 'decodewordspseudo'

    # expected words are the same for all subjects, so we can load them once here
    with open(folder1 + f'texts/{text}_1920/alltext.txt', encoding="utf-8") as f:
        expected_words = f.read().split()
    expected_words = expected_words[:24]
    
    worksheet.write_column(1, 0, expected_words)
    
    subs = [subno for subno in range(14501, 14514)]
    subs += [13512, 13514, 13515]
    
    # subs = [14504, 14508, 14505, 14502]
    
    # subs = [14504]

    # 3. Start collecting and writing data to the worksheet
    for sub_idx in tqdm(range(len(subs)), desc="Processing subjects"):

        sub_id = subs[sub_idx]
        worksheet.write(0, sub_idx + 1, sub_id)  # Write subject ID in the header row
        
        sub_path = os.path.join(folder1, folder2, f"subject_{sub_id}")
        if not os.path.exists(sub_path):
            continue
        
        print(f"\n{'='*55}")
        print(f"  Subject {sub_id}")
        print(f"{'='*55}")   
        
        detected_words_path = os.path.join(sub_path, f"detected_words_v3_{sub_id}.xlsx")
        if not os.path.exists(detected_words_path):
            print(f"  No detected words file found for subject {sub_id}. Skipping.")
            continue
            
        df = pd.read_excel(detected_words_path, usecols=[1])
        detected_words = df.iloc[:, 0].dropna().tolist()
        
        # Write detected words in the next column of the worksheet
        worksheet.write_column(1, sub_idx + 1, detected_words)
        
    # 4. Close the workbook to save it
    workbook.close()

collect_data_main()