import os


def get_directory_structure(root_dir, max_depth=3, min_depth=2, important_paths=None, exclude_dirs=None, exclude_patterns=None,
                            show_files=True):
    """
    Generate a clean directory structure with variable depth based on importance.

    Args:
        root_dir (str): The root directory to start from
        max_depth (int): Default maximum depth to traverse for important paths
        min_depth (int): Minimum depth to traverse for all paths
        important_paths (list): List of path fragments that should be explored more deeply
        exclude_dirs (list): Directories to exclude
        exclude_patterns (list): Patterns to exclude
        show_files (bool): Whether to show files or only directories

    Returns:
        str: Formatted directory structure
    """
    if important_paths is None:
        important_paths = ['models/portfolio', 'models/regime', 'core', 'analysis', 'execution']

    if exclude_dirs is None:
        exclude_dirs = ['venv', '__pycache__', '.git', '.idea', '.vscode', 'node_modules']

    if exclude_patterns is None:
        exclude_patterns = ['.pyc', '.pyo', '.DS_Store', '.gitignore', '.env']

    result = [f"Directory Structure for: {os.path.basename(root_dir)}\n"]

    def should_exclude(path):
        basename = os.path.basename(path)
        # Check if directory should be excluded
        if basename in exclude_dirs:
            return True

        # Check for excluded patterns
        for pattern in exclude_patterns:
            if pattern in basename:
                return True

        return False

    def is_important_path(path):
        rel_path = os.path.relpath(path, root_dir)
        for imp_path in important_paths:
            if imp_path in rel_path:
                return True
        return False

    def explore_directory(directory, prefix='', depth=0):
        # Determine depth limit for this directory
        # Important paths get explored to max_depth
        # All other paths get explored to at least min_depth
        current_max_depth = max_depth if is_important_path(directory) else min_depth

        if depth > current_max_depth:
            return

        # List and sort directory contents
        try:
            items = sorted(os.listdir(directory))
        except (PermissionError, FileNotFoundError):
            return

        # Process directories first, then files
        dirs = [item for item in items if
                os.path.isdir(os.path.join(directory, item)) and not should_exclude(os.path.join(directory, item))]

        if show_files:
            files = [item for item in items if os.path.isfile(os.path.join(directory, item)) and not should_exclude(
                os.path.join(directory, item))]
        else:
            files = []

        # Add directories to the result
        for i, dir_name in enumerate(dirs):
            dir_path = os.path.join(directory, dir_name)

            # Format directory name
            is_last_item = (i == len(dirs) - 1 and len(files) == 0)
            result.append(f"{prefix}{'└── ' if is_last_item else '├── '}{dir_name}/")

            # Recursively explore subdirectories
            explore_directory(
                dir_path,
                prefix=prefix + ('    ' if is_last_item else '│   '),
                depth=depth + 1
            )

        # Add files to the result
        for i, file_name in enumerate(files):
            is_last_file = (i == len(files) - 1)
            result.append(f"{prefix}{'└── ' if is_last_file else '├── '}{file_name}")

    explore_directory(root_dir)
    return '\n'.join(result)


if __name__ == "__main__":
    # Use the current directory as the starting point
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Define important paths that should be explored more deeply
    important_paths = [
        'models/portfolio',
        'models/regime',
        'models/strategies',
        'models/alpha',
        'models/research',
        'models/production',
        'core',
        'data/processors',
        'data/fetchers',
        'data/storage',
        'execution/order',
        'execution/exchange',
        'execution/fill',
        'execution/risk',
        'analysis',
        'visualization/panels',
        'visualization/adapters',
        'visualization/plots',
        'ui/web',
        'ui/api',
        'ui/components',
        'config'
    ]

    # Get the structure with standard depth but deeper for important paths
    structure = get_directory_structure(
        project_root,
        max_depth=4,       # Deeper exploration for important paths
        min_depth=2,       # Ensure all directories show at least some subdirectories
        important_paths=important_paths,
        exclude_dirs=['venv', '__pycache__', '.git', '.idea', 'node_modules'],
        exclude_patterns=['.pyc', '.pyo', '.DS_Store'],
        show_files=True    # Show both directories and files
    )

    print(structure)