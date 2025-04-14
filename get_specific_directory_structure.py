import os


def print_directory_structure(startpath, indent=0):
    """
    Print the directory structure starting from startpath (directories only)
    """
    print('|' + '-' * indent + os.path.basename(startpath) + '/')

    for item in sorted(os.listdir(startpath)):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print_directory_structure(path, indent + 4)


# Replace the path with your target directory
print_directory_structure('/Users/joshuahellerman/ML-Powered Live Trading Economy/core')