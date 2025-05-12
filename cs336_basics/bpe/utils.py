from pathlib import Path

def read_text_file(file_path: Path) -> str:
    """
    Read a text file and return its content as a string.
    
    Args:
        file_path (Path): Path to the text file.
        
    Returns:
        str: Content of the file.
    """
    with open(file_path, 'rb') as file:
        return file.read()