import os
import shutil
import logging

# Configure logging
logging.basicConfig(filename='deletion2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def deleteEmptyFiles():
    # Directory to scan for files
    directory = 'cache_raw_data_all_intraday'
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Check if it's a file
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                # print(f"Checking: {file_path}")
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                    
                    # Delete the file if it's empty or contains only 'null'
                    if content == '' or content == 'null':
                        os.remove(file_path)
                        logging.info(f"Deleted: {file_path}")
        
        # Remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                logging.info(f"Deleted empty directory: {dir_path}")

def delete5SDataFolders():
    # Directory to scan for files
    directory = 'cache2'
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            if dir.endswith('5S'):
                dir_path = os.path.join(root, dir)
                # shutil.rmtree(dir_path)
                logging.info(f"Deleted 5S data folder: {dir_path}")

# Call the function
# deleteEmptyFiles()
# delete5SDataFolders()
