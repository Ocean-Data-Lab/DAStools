
## Script transfer data from OOI Piweb server to ODL NAS

import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm


def list_files_in_directory(http_url):
    """List all files in the given HTTP directory and subdirectories."""
    response = requests.get(http_url)
    response.raise_for_status()  # Raise an error for bad responses

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    file_links = []
    folder_links = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        full_url = urljoin(http_url, href)

        if href.endswith('/'):  # It's a directory
            folder_links.append(full_url)
        else:  # It's a file
            file_links.append(full_url)
    
    return file_links, folder_links

def create_nested_directory(nas_base_path, http_url):
    """Create a nested directory structure on the NAS based on the HTTP URL."""
    relative_path = http_url.replace("http://", "").replace("https://", "").replace("/", os.sep)
    nas_path = os.path.join(nas_base_path, relative_path)

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(nas_path), exist_ok=True)

    return nas_path

def stream_file_to_nas(http_url, nas_path):
    """Stream a file from an HTTP server directly to the NAS with a progress bar."""
    response = requests.get(http_url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Get total file size
        file_exists = os.path.exists(nas_path)

        # If file exists, check its size to avoid re-downloading
        if file_exists and os.path.getsize(nas_path) == total_size:
            # print(f"File {os.path.basename(nas_path)} already exists and is fully downloaded.")
            return  # Skip the file

        with open(nas_path, 'wb') as f:
            # Initialize tqdm progress bar
            filename = os.path.basename(nas_path)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        # print(f"File successfully streamed to: {os.path.split(nas_path)[-2]}")
    else:
        print(f"Failed to download file. HTTP Status code: {response.status_code}")

def download_and_stream_files(http_url, nas_base_directory):
    # Get list of files and folders
    file_links, folder_links = list_files_in_directory(http_url)
    
    # Loop through each file link (if any)
    if file_links:
        folder_name = file_links[0].split('/')[-2]  # Get the folder name
        with tqdm(total=len(file_links), desc=f"Downloading {folder_name}") as progress_bar:
            for file_url in file_links:
                nas_path = create_nested_directory(nas_base_directory, file_url)  # Create necessary directories
                try:
                    # Stream the file directly to the NAS
                    stream_file_to_nas(file_url, nas_path)
                except Exception as e:
                    print(f"Failed to stream {file_url} to {nas_path}: {e}")
                progress_bar.update(1)

    # Recursively process each folder, until file_links is not empty anymore
    for folder_url in folder_links:
        download_and_stream_files(folder_url, nas_base_directory)  # Recursive call for subdirectories



# To mount the NAS as a local storage use, with write permissions, use the following commands:
# 
# ```bash
# sudo apt-get install cifs-utils
# sudo mkdir -p /media/odl_nas
# sudo mount -t cifs -o credentials=/home/<user>/.smbcredentials,dir_mode=0777,file_mode=0777 //odl.ocean.washington.edu/ODL /media/odl_nas
# ```
# 
# After creating `.smbcredentials` with this format:
# 
# ```bash
# username=your_username
# password=your_password
# ```

if __name__ == "__main__":
    # Example usage:
    http_url = "http://piweb.ooirsn.uw.edu/das/"
    nas_directory = "/media/odl_nas/ODLdata/ooiDAS/"
    # Stream the file directly to NAS
    download_and_stream_files(http_url, nas_directory)


