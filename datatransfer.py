
## Script transfer data from OOI Piweb server to ODL NAS

import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


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


def get_request_with_retry(url, retries=3, backoff_factor=0.3):
    """Get a request with retries and backoff for robustness against network issues."""
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session.get(url, stream=True)

def stream_file_to_nas(http_url, nas_path):
    """Stream a file from an HTTP server directly to the NAS with retries and improved path handling."""
    response = get_request_with_retry(http_url)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Get total file size
        nas_path = Path(nas_path)
        
        # If file exists and size matches, skip the download
        if nas_path.exists() and nas_path.stat().st_size == total_size:
            return  # Skip the file

        # Stream and save the file
        with open(nas_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):  # Larger chunk size for faster transfer
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
        print(f"File successfully streamed to: {nas_path.parent}")
    else:
        print(f"Failed to download file {http_url}. HTTP Status code: {response.status_code}")


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


