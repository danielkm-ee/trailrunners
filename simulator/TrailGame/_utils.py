import os

def find_path(pathstr, startpath, filename):
    # Scans files from current working directory recursively until a file called 'FILENAME' is found. Otherwise NoneType is returned
    with os.scandir(startpath) as it:
        for entry in it:
            if not entry.name.startswith('.') and not entry.is_file():
                pathstr = os.path.join(pathstr, entry.name)
                return find_path(pathstr, pathstr, filename)
            if entry.name == filename:
                pathstr = os.path.join(pathstr, entry.name)
                return pathstr
