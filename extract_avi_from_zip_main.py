import subprocess
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from globalVars import folder1, folder2

def extract_avi_from_zip(zip_path, output_dir, password=None, seven_zip_path=r"C:\Program Files\7-Zip\7z.exe"):
    """
    Extract all .avi files from a zip archive using 7-Zip (much faster than Python zipfile).
    If .avi files already exist, they are temporarily renamed, then restored at the end.
    
    Parameters:
    -----------
    zip_path : str or Path
        Path to the zip file
    output_dir : str or Path
        Directory where .avi files will be extracted
    password : str, optional
        Password for encrypted zip files
    seven_zip_path : str
        Path to 7z.exe executable
    
    Returns:
    --------
    list
        List of paths to extracted .avi files
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if 7-Zip exists
    if not os.path.exists(seven_zip_path):
        raise FileNotFoundError(
            f"7-Zip not found at {seven_zip_path}\n"
            "Please install 7-Zip from https://www.7-zip.org/ or update the path"
        )
    
    # Check if .avi files already exist and rename them temporarily
    existing_avis = [f for f in output_dir.glob("*.avi") if "_backup_" not in f.name]
    backup_mapping = {}  # Store original -> backup mapping
    
    if existing_avis:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"⚠️  Found {len(existing_avis)} existing .avi file(s), temporarily renaming...")
        for avi_file in existing_avis:
            backup_name = avi_file.stem + f"_backup_{timestamp}" + avi_file.suffix
            backup_path = avi_file.parent / backup_name
            avi_file.rename(backup_path)
            backup_mapping[backup_path] = avi_file  # Store for restoration
            print(f"  Renamed: {avi_file.name} → {backup_name}")
    
    # Build 7-Zip command
    cmd = [
        seven_zip_path,
        "e",  # Extract without directory structure
        str(zip_path),
        f"-o{output_dir}",  # Output directory
        "*.avi",  # Only extract .avi files
        "-aoa",  # Overwrite all existing files without prompt
        "-bsp1"  # Show progress percentage
    ]
    
    # Add password if provided
    if password:
        cmd.append(f"-p{password}")
    
    print(f"Extracting .avi files from {zip_path.name}...")
    
    # Run 7-Zip
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'  # Handle non-UTF8 characters in filenames
        )
        
        if result.returncode == 0:
            # Find extracted .avi files (excluding backup files that may have been in the zip)
            extracted_files = [f for f in output_dir.glob("*.avi") if "_backup_" not in f.name]
            
            print(f"✓ Extracted {len(extracted_files)} .avi file(s)")
            for avi_file in extracted_files:
                print(f"  → {avi_file.name}")
            
            # Restore backup files to their original names
            if backup_mapping:
                print(f"\n🔄 Restoring {len(backup_mapping)} backup file(s) to original names...")
                for backup_path, original_path in backup_mapping.items():
                    if backup_path.exists():
                        # Check if original_path already exists (extracted from zip)
                        if original_path.exists():
                            # The zip contained a newer version, delete the old backup
                            print(f"  Skipping restore (new version extracted): {original_path.name}")
                            backup_path.unlink()  # Delete the old backup
                        else:
                            # No new version extracted, restore the backup
                            backup_path.rename(original_path)
                            print(f"  Restored: {backup_path.name} → {original_path.name}")
            
            return [str(f) for f in extracted_files]
        
        elif result.returncode == 2:
            # Restore backups even on error
            _restore_backups(backup_mapping)
            
            # Password error or file not found
            if "Wrong password" in result.stderr or "Can not open encrypted archive" in result.stderr:
                raise RuntimeError(f"Incorrect password for {zip_path.name}")
            else:
                raise RuntimeError(f"7-Zip error: {result.stderr}")
        
        else:
            # Restore backups even on error
            _restore_backups(backup_mapping)
            raise RuntimeError(f"7-Zip failed with code {result.returncode}: {result.stderr}")
    
    except FileNotFoundError:
        # Restore backups on exception
        _restore_backups(backup_mapping)
        raise FileNotFoundError(
            f"Could not execute 7-Zip at {seven_zip_path}\n"
            "Make sure 7-Zip is installed correctly"
        )
    
    except Exception as e:
        # Restore backups on any exception
        _restore_backups(backup_mapping)
        raise


def _restore_backups(backup_mapping):
    """Helper function to restore backup files."""
    if backup_mapping:
        print(f"\n🔄 Restoring backup files...")
        for backup_path, original_path in backup_mapping.items():
            if backup_path.exists():
                if not original_path.exists():
                    backup_path.rename(original_path)
                    print(f"  Restored: {backup_path.name} → {original_path.name}")
                else:
                    print(f"  Skipping (original exists): {original_path.name}")


# Example usage
if __name__ == "__main__":
    text = 'decodewordspseudo'
    
    subs = [subno for subno in range(13703, 13901)]  # Multiple subjects for testing
    
    for sub_id in tqdm(subs, desc="Processing subjects"):
        # Example 1: Extract from a single zip file
        sub_path = os.path.join(folder1, folder2, f"subject_{sub_id}")
        if not os.path.exists(sub_path):
            continue
        
        text_folder_name = [f for f in os.listdir(sub_path)
            if text in f and not f.endswith(".avi") and not f.endswith(".wav") 
            and not f.endswith(".txt") and not f.endswith(".json") 
            and not f.endswith(".xlsx") and 'clips' not in f]
        
        if not text_folder_name:
            continue
            
        text_folder_path = os.path.join(sub_path, text_folder_name[0])
        
        text_path_name = [f for f in os.listdir(text_folder_path) 
                          if text in f and f.endswith(".zip")]
        
        if not text_path_name:
            print(f'No .zip file found for sub = {sub_id}, in folder = {text_folder_path}')
            continue
        
        final_text_path = Path(text_folder_path) / text_path_name[0]
        
        try:
            extracted = extract_avi_from_zip(
                final_text_path, 
                output_dir=text_folder_path, 
                password='2020r@d@r'
            )
        except Exception as e:
            print(f'❌ Error processing {final_text_path.name}: {e}')