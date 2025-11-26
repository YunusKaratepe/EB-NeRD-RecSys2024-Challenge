import yaml
import os

def read_yaml(path):
    with open(path, 'r') as file:
        content = yaml.safe_load(file)
    return content

def get_feature_folders(folder):
    """
    Returns a dictionary of parquet files found in the folder.
    Keys are tuples of (feature_name, size, split).
    Values are relative paths to the parquet files.
    
    Example:
        If folder contains: feature-extraction/output/features/a_additional_feature/small/test_feat.parquet
        Returns: {('a_additional_feature', 'small', 'test'): 'a_additional_feature/small/test_feat.parquet'}
    """
    result = {}
    
    for root, dirs, files in os.walk(folder):
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        if parquet_files:
            # Extract the relative path from the input folder
            rel_path = os.path.relpath(root, folder)
            path_parts = rel_path.split(os.sep)
            
            # Expecting structure: feature_name/size/file.parquet
            if len(path_parts) >= 2:
                feature_name = path_parts[0]
                size = path_parts[1]
                
                # Determine split (train, test, validation) from filename
                for parquet_file in parquet_files:
                    full_path = os.path.join(root, parquet_file).replace(os.sep, '/')
                    
                    # Extract split from filename
                    if 'train' in parquet_file.lower():
                        split = 'train'
                    elif 'test' in parquet_file.lower():
                        split = 'test'
                    elif 'validation' in parquet_file.lower() or 'val' in parquet_file.lower():
                        split = 'validation'
                    else:
                        split = 'unknown'
                    
                    key = (feature_name, size, split)
                    result[key] = full_path
    
    return result