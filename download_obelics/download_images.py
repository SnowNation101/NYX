import requests
import os

def download_image_from_url(image_url: str, save_path: str, timeout: int = 10):
    """
    Downloads an image from the given URL.
    Prints an error message if the image does not exist (404 error)
    or if the download does not complete within the specified timeout.

    Args:
        image_url (str): The URL of the image to download.
        save_path (str): The local path where the image will be saved,
                         including the filename and extension.
        timeout (int): The download timeout in seconds. If the download
                       exceeds this time, an error will be raised.
    """
    try:
        # Send a GET request to the image URL with a specified timeout.
        # stream=True allows downloading large files in chunks without loading
        # the entire content into memory at once.
        response = requests.get(image_url, stream=True, timeout=timeout)
        
        # Raise an HTTPError for bad responses (4xx or 5xx status codes).
        # This will catch 404 errors automatically.
        response.raise_for_status() 

        # Ensure the directory for saving the image exists.
        # If it doesn't exist, it will be created. exist_ok=True prevents an
        # error if the directory already exists.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Open the specified file in binary write mode ('wb') and write the
        # image content in chunks.
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image successfully downloaded to: {save_path}")

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors, specifically checking for 404 Not Found.
        if e.response.status_code == 404:
            print(f"Error: Image not found or invalid URL (404 Not Found) - {image_url}")
        else:
            print(f"An HTTP error occurred during download: {e}")
    except requests.exceptions.ConnectionError as e:
        # Handle network connection errors.
        print(f"Error: Could not connect to the server. Please check your network connection or URL - {e}")
    except requests.exceptions.Timeout as e:
        # Handle download timeout errors.
        print(f"Error: Download timed out. Exceeded {timeout} seconds - {e}")
    except requests.exceptions.RequestException as e:
        # Handle any other requests-related errors.
        print(f"An unknown error occurred during download: {e}")
    except Exception as e:
        # Handle any other unexpected errors.
        print(f"An unexpected error occurred: {e}")

# --- How to Use ---
if __name__ == "__main__":
    # Example usage:
    # Replace the URLs and save paths with your desired values for testing.
    
    # Example of an existing image URL (please replace with a real, accessible URL)
    # This URL for a Google logo might become invalid, so it's good to have a backup.
    existing_image_url = "https://bpb-eu-w2.wpmucdn.com/blogs.reading.ac.uk/dist/9/82/files/2015/12/Mahleb2.jpg" 
    
    save_directory = "./downloaded_images"
    existing_image_save_path = os.path.join(save_directory, "sample_image.jpg")

    print("--- Attempting to download an existing image (default timeout 10s) ---")
    download_image_from_url(existing_image_url, existing_image_save_path)
    print("\n")

    # Example to simulate a slow download or trigger a timeout.
    # To truly test a timeout, you might need a very slow server or a large file.
    # We'll set a very short timeout here to easily trigger the error.
    slow_image_url = "https://speed.hetzner.de/100MB.bin" # A larger test file URL
    slow_image_save_path = os.path.join(save_directory, "slow_file.bin")
    
    print("--- Attempting to download a potentially slow image (timeout set to 2s) ---")
    # Setting a 2-second timeout makes it easy to trigger if the file is large
    # or the network is slow.
    download_image_from_url(slow_image_url, slow_image_save_path, timeout=2) 
    print("\n")

    # Example of a non-existent image URL (expected 404 error)
    non_existent_image_url = "http://example.com/non_existent_image.jpg"
    non_existent_image_save_path = os.path.join(save_directory, "non_existent_image.jpg")

    print("--- Attempting to download a non-existent image (expected 404 error) ---")
    download_image_from_url(non_existent_image_url, non_existent_image_save_path)
    print("\n")

    # Example of an invalid URL format
    invalid_url = "this is not a valid URL"
    invalid_save_path = os.path.join(save_directory, "invalid_url_image.jpg")
    print("--- Attempting to use an invalid URL ---")
    download_image_from_url(invalid_url, invalid_save_path)