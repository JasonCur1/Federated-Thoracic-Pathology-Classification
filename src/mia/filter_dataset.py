import glob
import mia_utils
from pathlib import Path
import pandas as pd
from baseline.config import Config, CFG


ORIG_PATH = Path(r'C:\Users\sofia\.cache\huggingface\hub\datasets--exalsius--NIH-Chest-XRay-Federated\snapshots\7e09ef5d315c088a5a26eb68d008eafcd9111c7b')
OUTPUT_PATH = CFG.project_root / 'mia_data'

HOSPITALS = ['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']

def get_image_paths_set(dm):
    image_paths = set()

    for df in [dm.train_df, dm.val_df, dm.test_df, dm.ood_df]:
        for img in df['image']:
            image_paths.add(img['path'])
    return image_paths
    
def filter_and_save_hospital(data_root, hospital, keep_paths, output_root):
    pattern = str(data_root / hospital / '**' / '*.parquet')
    paths = glob.glob(pattern, recursive=True)

    if not paths:
        print(f'No files for {hospital}')
        return

    for p in paths:
        df = pd.read_parquet(p)
        
        img_paths = df['image'].map(lambda x: x['path'])
        filtered_df = df[img_paths.isin(keep_paths)]

        if len(filtered_df) == 0:
            continue

        rel_path = Path(p).relative_to(data_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        filtered_df.to_parquet(out_path)
        print(f'{hospital}: kept {len(filtered_df)} rows from {p}')

def main():
    if OUTPUT_PATH.exists() and any(OUTPUT_PATH.iterdir()):
        raise RuntimeError(f'{OUTPUT_PATH} is not empty. Aborting to avoid overwrite.')
        
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    downsampled_cfg = CFG
    original_cfg = Config(data_root_path=ORIG_PATH)

    print('Loading datamodules...')
    downsampled_dm = mia_utils.load_data(downsampled_cfg)
    original_dm = mia_utils.load_data(original_cfg)
    print('Datamodules loaded.')

    print('\nBuilding image path sets...')
    downsampled_paths_set = get_image_paths_set(downsampled_dm)
    original_paths_set = get_image_paths_set(original_dm)
    
    print(f'Downsampled: {len(downsampled_paths_set)} images')
    print(f'Original: {len(original_paths_set)} images')

    mia_paths_set = original_paths_set - downsampled_paths_set
    print(f'MIA set (difference): {len(mia_paths_set)} images')

    print('\nFiltering + saving hospital data for MIA...')
    for hospital in HOSPITALS:
        filter_and_save_hospital(
            data_root=original_cfg.data_root,
            hospital=hospital,
            keep_paths=mia_paths_set,
            output_root=OUTPUT_PATH,
        )
    print('MIA dataset saved.')

    mia_cfg = Config(data_root_path=OUTPUT_PATH)
    mia_dm = mia_utils.load_data(mia_cfg)
    mia_paths_set = get_image_paths_set(mia_dm)

    intersection = mia_paths_set & downsampled_paths_set
    
    if len(intersection) != 0:
        print('WARNING: overlap detected.')
        print(list(intersection)[:10])
    else:
        print('No overlap between MIA and downsampled sets.')
    
    print('Done!') 
    # trim.py @ 0.3 after this (~31GB -> ~10GB)

if __name__ == '__main__':
    main()

