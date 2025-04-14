import os
import re
from collections import defaultdict


def analyze_dependencies(root_dir):
    """
    Analyze Python file dependencies in the given directory.

    Args:
        root_dir: Root directory of the project

    Returns:
        dict: Mapping of module to its dependencies
    """
    dependencies = defaultdict(list)
    python_files = []

    # Find all Python files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                python_files.append(rel_path)

    # Extract import statements from each file
    for file_path in python_files:
        module_name = file_path[:-3].replace('/', '.').replace('\\', '.')

        with open(os.path.join(root_dir, file_path), 'r', encoding='utf-8') as f:
            try:
                content = f.read()

                # Regular expressions to match different import patterns
                # Standard import
                standard_imports = re.findall(r'import\s+([\w\.]+)', content)
                # From import
                from_imports = re.findall(r'from\s+([\w\.]+)\s+import', content)

                # Combine all imports
                all_imports = standard_imports + from_imports

                # Filter imports to only include project-specific ones
                project_imports = [imp for imp in all_imports if any(imp.startswith(prefix) for prefix in
                                                                     ['core', 'data', 'models', 'execution',
                                                                      'visualization', 'ui', 'analysis', 'utils'])]

                dependencies[module_name] = project_imports
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return dependencies


def format_dependency_map(dependencies):
    """
    Format the dependency map as a readable markdown text.

    Args:
        dependencies: Dict mapping modules to their dependencies

    Returns:
        str: Formatted dependency map
    """
    formatted_map = ""

    # Group by top-level module
    grouped_modules = defaultdict(list)
    for module in sorted(dependencies.keys()):
        parts = module.split('.')
        if len(parts) > 0:
            grouped_modules[parts[0]].append(module)

    # Format each group
    for group_name, modules in sorted(grouped_modules.items()):
        formatted_map += f"**{group_name.upper()}:**\n"

        for module in sorted(modules):
            deps = dependencies[module]
            formatted_map += f"* **{module}**\n"
            if deps:
                formatted_map += f"   * Depends on: {', '.join(sorted(deps))}\n"

        formatted_map += "\n"

    return formatted_map


def main():
    # Replace with your project's root directory
    root_dir = "."  # Current directory, adjust as needed

    # Analyze current dependencies
    current_dependencies = analyze_dependencies(root_dir)

    # Format the dependency map
    formatted_map = format_dependency_map(current_dependencies)

    # Save to file
    with open('current_dependency_map.md', 'w', encoding='utf-8') as f:
        f.write(formatted_map)

    print(f"Dependency map saved to current_dependency_map.md")
    print("Here's the dependency map:")
    print("--------------------------")
    print(formatted_map)


if __name__ == "__main__":
    main()