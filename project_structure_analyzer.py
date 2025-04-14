#!/usr/bin/env python3
"""
project_structure_analyzer.py

A specialized script to analyze the structure of the ML-powered trading system,
with particular focus on visualization components and UI integration.

This script examines:
1. Module dependencies and import relationships
2. File completeness and development status
3. Component integration points
4. Missing or underdeveloped components
5. Structural inconsistencies

Run this script from the project root directory.
"""

import os
import re
import sys
import ast
import json
import importlib
from pathlib import Path
from collections import defaultdict, Counter


class ProjectStructureAnalyzer:
    """Analyzes the structure of the ML-powered trading system project."""

    def __init__(self, project_root=None, output_file=None):
        """Initialize the project structure analyzer."""
        self.project_root = project_root or os.getcwd()
        self.output_file = output_file or "project_structure_analysis.json"

        # Analysis data structures
        self.files = {}  # All project files
        self.modules = {}  # Python modules
        self.imports = defaultdict(set)  # Import relationships
        self.dependencies = defaultdict(set)  # Module dependencies
        self.reverse_dependencies = defaultdict(set)  # Module reverse dependencies
        self.component_status = {}  # Component development status
        self.visualization_components = []  # Visualization components
        self.ui_components = []  # UI components
        self.integration_points = []  # Integration points
        self.issues = []  # Issues found
        self.todo_markers = {}  # TODO markers in code
        self.underdeveloped_modules = []  # Modules that appear underdeveloped

        # Analysis flags
        self.incomplete_files = []
        self.syntax_errors = []
        self.potential_misalignments = []

        # Configuration directories to analyze
        self.key_directories = [
            'visualization',
            'ui',
            'core',
            'data',
            'models',
            'strategies',
            'execution',
            'utils'
        ]

    def analyze(self):
        """Run the full project structure analysis."""
        print(f"Analyzing project structure at: {self.project_root}")

        # Collect all project files
        self._collect_files()

        # Analyze Python files
        self._analyze_python_files()

        # Analyze import relationships
        self._analyze_import_relationships()

        # Check component development status
        self._check_component_status()

        # Find integration points
        self._find_integration_points()

        # Identify issues
        self._identify_issues()

        # Generate final report
        report = self._generate_report()

        # Print summary
        self._print_summary(report)

        # Save report to file
        self._save_report(report)

        return report

    def _collect_files(self):
        """Collect all project files and organize by type."""
        print("Collecting files...")

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden dirs and virtual environments
            dirs[:] = [d for d in dirs if not d.startswith('.') and
                       d not in ['venv', '__pycache__', 'env', '.git', '.idea']]

            rel_path = os.path.relpath(root, self.project_root)
            if rel_path == '.':
                rel_path = ''

            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                rel_file_path = os.path.join(rel_path, file)

                # Get file extension
                _, ext = os.path.splitext(file)
                ext = ext.lower()

                # Check if this is in a key directory
                in_key_dir = any(rel_path.startswith(d) for d in self.key_directories)

                # Store file info
                self.files[rel_file_path] = {
                    'path': file_path,
                    'relative_path': rel_file_path,
                    'extension': ext,
                    'size': os.path.getsize(file_path),
                    'in_key_directory': in_key_dir,
                }

                # Classify by directory
                if 'visualization' in rel_path:
                    self.visualization_components.append(rel_file_path)
                if 'ui' in rel_path:
                    self.ui_components.append(rel_file_path)

                # Check for todo markers, incomplete files
                if ext in ['.py', '.js', '.html', '.css', '.md', '.txt']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                            # Look for TODO markers
                            todo_matches = re.findall(r'(?i)TODO[:\s]*(.*?)(?:\n|$)', content)
                            if todo_matches:
                                self.todo_markers[rel_file_path] = [m.strip() for m in todo_matches]

                            # Check for file completeness heuristics
                            if "not implemented" in content.lower() or "TODO" in content:
                                self.incomplete_files.append(rel_file_path)

                            # Check if file appears to be a template or skeleton
                            if ext == '.py' and "pass" in content and len(content.strip().split('\n')) < 20:
                                self.incomplete_files.append(rel_file_path)

                    except Exception as e:
                        print(f"Error reading {rel_file_path}: {str(e)}")

    def _analyze_python_files(self):
        """Analyze Python files for imports, classes, functions, etc."""
        print("Analyzing Python files...")

        for rel_path, file_info in self.files.items():
            if file_info['extension'] != '.py':
                continue

            file_path = file_info['path']
            module_path = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')

            # Analyze Python module
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse the file using AST
                try:
                    tree = ast.parse(content)

                    # Extract imports
                    imports = self._extract_imports(tree)
                    self.imports[module_path] = imports

                    # Extract module info
                    module_info = self._extract_module_info(tree, content)
                    self.modules[module_path] = module_info

                    # Check for underdeveloped modules
                    if self._is_underdeveloped(module_info, content):
                        self.underdeveloped_modules.append(module_path)

                except SyntaxError as e:
                    print(f"Syntax error in {rel_path}: {str(e)}")
                    self.syntax_errors.append(rel_path)

            except Exception as e:
                print(f"Error processing {rel_path}: {str(e)}")

    def _extract_imports(self, tree):
        """Extract imports from an AST tree."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for name in node.names:
                        imports.add(f"{node.module}.{name.name}")

        return imports

    def _extract_module_info(self, tree, content):
        """Extract module information from an AST tree."""
        classes = []
        functions = []
        doc_string = ast.get_docstring(tree)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'doc': ast.get_docstring(node)
                }
                classes.append(class_info)
            elif isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'doc': ast.get_docstring(node)
                }
                functions.append(func_info)

        # Calculate code metrics
        lines = content.split('\n')
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        blank_lines = len([l for l in lines if not l.strip()])

        return {
            'classes': classes,
            'functions': functions,
            'doc_string': doc_string,
            'metrics': {
                'total_lines': len(lines),
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'class_count': len(classes),
                'function_count': len(functions)
            }
        }

    def _is_underdeveloped(self, module_info, content):
        """Check if a module appears to be underdeveloped based on heuristics."""
        metrics = module_info['metrics']

        # Heuristics for underdeveloped modules
        if metrics['total_lines'] < 20:
            return True

        if metrics['class_count'] == 0 and metrics['function_count'] == 0:
            return True

        if "not implemented" in content.lower() or "TODO" in content:
            return True

        if metrics['code_lines'] < 10 and "pass" in content:
            return True

        return False

    def _analyze_import_relationships(self):
        """Analyze import relationships to build dependency graph."""
        print("Analyzing import relationships...")

        # Build direct dependencies
        for module, imports in self.imports.items():
            for imp in imports:
                # Check if this is an internal project import
                if imp in self.modules or any(imp.startswith(m + '.') for m in self.modules):
                    # Find the base module
                    base_module = imp.split('.')[0]
                    if base_module in self.key_directories:
                        # This is a project module import
                        self.dependencies[module].add(imp)
                        self.reverse_dependencies[imp].add(module)

        # Calculate dependency counts for each module
        for module in self.modules:
            outgoing = len(self.dependencies[module])
            incoming = len(self.reverse_dependencies[module])

            self.modules[module]['dependencies'] = {
                'outgoing': outgoing,
                'incoming': incoming,
                'total': outgoing + incoming
            }

    def _check_component_status(self):
        """Check the development status of key components."""
        print("Checking component status...")

        # Define key component categories
        categories = {
            'visualization/dashboard': 'Dashboard backend',
            'visualization/plots': 'Visualization plots',
            'ui/web': 'Web interface',
            'ui/api': 'API interfaces',
        }

        # Check each category
        for path, description in categories.items():
            files = [f for f in self.files if f.startswith(path)]

            if not files:
                self.component_status[description] = 'missing'
                continue

            # Check if the component is underdeveloped
            module_paths = [f.replace('/', '.').replace('\\', '.').replace('.py', '')
                            for f in files if f.endswith('.py')]

            underdeveloped = [m for m in module_paths if m in self.underdeveloped_modules]
            incomplete = [f for f in files if f in self.incomplete_files]

            if len(underdeveloped) / max(1, len(module_paths)) > 0.5:
                # More than half of modules are underdeveloped
                self.component_status[description] = 'underdeveloped'
            elif incomplete:
                # Some files are marked as incomplete
                self.component_status[description] = 'incomplete'
            else:
                # Seems reasonably developed
                self.component_status[description] = 'developed'

            # Store details
            self.component_status[f"{description}_details"] = {
                'files': files,
                'module_paths': module_paths,
                'underdeveloped': underdeveloped,
                'incomplete': incomplete
            }

    def _find_integration_points(self):
        """Find integration points between components."""
        print("Finding integration points...")

        # Check for imports between visualization and UI
        for module, imports in self.imports.items():
            # Check if this is a visualization module
            if module.startswith('visualization.'):
                # Look for UI imports
                ui_imports = [imp for imp in imports if imp.startswith('ui.')]
                if ui_imports:
                    self.integration_points.append({
                        'type': 'visualization_importing_ui',
                        'module': module,
                        'imports': list(ui_imports)
                    })

            # Check if this is a UI module
            if module.startswith('ui.'):
                # Look for visualization imports
                vis_imports = [imp for imp in imports if imp.startswith('visualization.')]
                if vis_imports:
                    self.integration_points.append({
                        'type': 'ui_importing_visualization',
                        'module': module,
                        'imports': list(vis_imports)
                    })

        # Check for potential middleware or adapter modules
        for module in self.modules:
            if 'adapter' in module.lower() or 'connector' in module.lower() or 'bridge' in module.lower():
                # This might be an integration module
                vis_imports = [imp for imp in self.imports.get(module, [])
                               if imp.startswith('visualization.')]
                ui_imports = [imp for imp in self.imports.get(module, [])
                              if imp.startswith('ui.')]

                if vis_imports and ui_imports:
                    self.integration_points.append({
                        'type': 'adapter_module',
                        'module': module,
                        'visualization_imports': list(vis_imports),
                        'ui_imports': list(ui_imports)
                    })

    def _identify_issues(self):
        """Identify structural issues based on analysis."""
        print("Identifying issues...")

        # Check for incompletely developed components
        for component, status in self.component_status.items():
            if not component.endswith('_details') and status in ['missing', 'underdeveloped', 'incomplete']:
                details = self.component_status.get(f"{component}_details", {})
                self.issues.append({
                    'type': f"{status}_component",
                    'component': component,
                    'details': details,
                    'description': f"The {component} component is {status}."
                })

        # Check for missing integration between visualization and UI
        if not any(p['type'] in ['ui_importing_visualization', 'visualization_importing_ui', 'adapter_module']
                   for p in self.integration_points):
            self.issues.append({
                'type': 'missing_integration',
                'description': "No clear integration found between visualization and UI components."
            })

        # Check for modules with syntax errors
        if self.syntax_errors:
            self.issues.append({
                'type': 'syntax_errors',
                'files': self.syntax_errors,
                'description': f"Found {len(self.syntax_errors)} files with syntax errors."
            })

        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies()
        if circular_deps:
            self.issues.append({
                'type': 'circular_dependencies',
                'dependencies': circular_deps,
                'description': f"Found {len(circular_deps)} circular dependencies between modules."
            })

        # Check for inconsistent imports
        inconsistent_imports = self._find_inconsistent_imports()
        if inconsistent_imports:
            self.issues.append({
                'type': 'inconsistent_imports',
                'imports': inconsistent_imports,
                'description': f"Found {len(inconsistent_imports)} potentially inconsistent imports."
            })

        # Check for potential misalignment between visualization and UI
        if self.visualization_components and self.ui_components:
            # Check for matching file patterns
            vis_patterns = set(self._extract_component_names(self.visualization_components))
            ui_patterns = set(self._extract_component_names(self.ui_components))

            # Find patterns in visualization not in UI
            vis_not_in_ui = vis_patterns - ui_patterns
            ui_not_in_vis = ui_patterns - vis_patterns

            if vis_not_in_ui:
                self.issues.append({
                    'type': 'misaligned_components',
                    'direction': 'visualization_missing_in_ui',
                    'components': list(vis_not_in_ui),
                    'description': f"Found {len(vis_not_in_ui)} components in visualization without matching UI components."
                })

            if ui_not_in_vis:
                self.issues.append({
                    'type': 'misaligned_components',
                    'direction': 'ui_missing_in_visualization',
                    'components': list(ui_not_in_vis),
                    'description': f"Found {len(ui_not_in_vis)} components in UI without matching visualization components."
                })

    def _find_circular_dependencies(self):
        """Find circular dependencies in the module graph."""
        circular = []

        def dfs(node, path=None):
            if path is None:
                path = []

            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular.append(cycle)
                return

            for dep in self.dependencies.get(node, []):
                # Only consider module level dependencies
                base_dep = dep.split('.')[0]
                if base_dep in self.modules:
                    dfs(base_dep, path + [node])

        for module in self.modules:
            base_module = module.split('.')[0]
            dfs(base_module)

        return circular

    def _find_inconsistent_imports(self):
        """Find potentially inconsistent imports."""
        inconsistent = []

        # Check for relative vs absolute imports
        for module, imports in self.imports.items():
            module_parts = module.split('.')

            for imp in imports:
                # Check if this import could be from the same package
                imp_parts = imp.split('.')

                if len(module_parts) > 1 and len(imp_parts) > 1:
                    if module_parts[0] == imp_parts[0]:
                        # This is an import from the same top-level package
                        # Check if it's inconsistently using absolute imports

                        # Find other imports from the same module
                        for other_module, other_imports in self.imports.items():
                            if other_module != module and other_module.split('.')[0] == module_parts[0]:
                                # Same top-level package
                                for other_imp in other_imports:
                                    if other_imp.split('.')[0] == imp_parts[0]:
                                        # Found another import from the same package
                                        if '.' in other_imp and '.' in imp and other_imp != imp:
                                            # Different import style
                                            inconsistent.append({
                                                'module1': module,
                                                'import1': imp,
                                                'module2': other_module,
                                                'import2': other_imp
                                            })

        return inconsistent

    def _extract_component_names(self, file_paths):
        """Extract component names from file paths for matching."""
        component_names = []

        # Extract core component names without directories or extensions
        for path in file_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            # Remove common prefixes/suffixes
            name = re.sub(r'(_component|_panel|_view|_controller|_service)$', '', name)
            component_names.append(name)

        return component_names

    def _generate_report(self):
        """Generate a comprehensive report of the project structure analysis."""
        print("Generating report...")

        # Count files by directory
        dir_counts = Counter()
        for file_path in self.files:
            dir_name = os.path.dirname(file_path)
            if not dir_name:
                dir_name = 'root'
            dir_counts[dir_name] += 1

        # Count files by extension
        ext_counts = Counter()
        for file_info in self.files.values():
            ext = file_info['extension']
            ext_counts[ext] += 1

        # Calculate component statistics
        vis_count = len(self.visualization_components)
        ui_count = len(self.ui_components)
        integration_count = len(self.integration_points)
        issue_count = len(self.issues)

        # Collect key areas of concern
        areas_of_concern = []
        for issue in self.issues:
            areas_of_concern.append({
                'type': issue['type'],
                'description': issue['description']
            })

        # List all potentially incomplete files
        incomplete_files = []
        for file_path in self.incomplete_files:
            if file_path in self.todo_markers:
                todos = self.todo_markers[file_path]
            else:
                todos = []

            incomplete_files.append({
                'file': file_path,
                'todos': todos
            })

        # Generate structure tree
        structure_tree = self._generate_structure_tree()

        # Generate module dependency graph
        dependency_graph = self._generate_dependency_graph()

        return {
            'project_root': self.project_root,
            'summary': {
                'total_files': len(self.files),
                'python_modules': len(self.modules),
                'visualization_components': vis_count,
                'ui_components': ui_count,
                'integration_points': integration_count,
                'issues_found': issue_count,
                'underdeveloped_modules': len(self.underdeveloped_modules),
                'incomplete_files': len(self.incomplete_files),
                'syntax_errors': len(self.syntax_errors)
            },
            'structure': {
                'directories': dict(dir_counts),
                'extensions': dict(ext_counts),
                'tree': structure_tree
            },
            'components': {
                'status': {k: v for k, v in self.component_status.items() if not k.endswith('_details')},
                'visualization': self.visualization_components,
                'ui': self.ui_components
            },
            'integration': {
                'points': self.integration_points,
            },
            'issues': {
                'list': self.issues,
                'areas_of_concern': areas_of_concern,
                'incomplete_files': incomplete_files,
                'syntax_errors': self.syntax_errors,
                'underdeveloped_modules': self.underdeveloped_modules
            },
            'dependencies': {
                'graph': dependency_graph
            }
        }

    def _generate_structure_tree(self):
        """Generate a tree representation of the project structure."""
        tree = {}

        for file_path in sorted(self.files.keys()):
            path_parts = file_path.split(os.sep)

            current = tree
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # This is a file
                    if '__files__' not in current:
                        current['__files__'] = []
                    current['__files__'].append(part)
                else:
                    # This is a directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        return tree

    def _generate_dependency_graph(self):
        """Generate a graph representation of module dependencies."""
        graph = {
            'nodes': [],
            'edges': []
        }

        # Add nodes for each module
        for module in self.modules:
            base_module = module.split('.')[0]
            if base_module in ['visualization', 'ui']:
                # Only include visualization and UI modules in the graph
                graph['nodes'].append({
                    'id': module,
                    'group': base_module
                })

        # Add edges for dependencies
        for module, deps in self.dependencies.items():
            base_module = module.split('.')[0]
            if base_module not in ['visualization', 'ui']:
                continue

            for dep in deps:
                dep_base = dep.split('.')[0]
                if dep_base in ['visualization', 'ui']:
                    graph['edges'].append({
                        'source': module,
                        'target': dep
                    })

        return graph

    def _print_summary(self, report):
        """Print a summary of the analysis results."""
        print("\n" + "=" * 60)
        print(" PROJECT STRUCTURE ANALYSIS SUMMARY ")
        print("=" * 60)

        summary = report['summary']

        print(f"\nProject Root: {self.project_root}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Python Modules: {summary['python_modules']}")
        print(f"Visualization Components: {summary['visualization_components']}")
        print(f"UI Components: {summary['ui_components']}")
        print(f"Integration Points: {summary['integration_points']}")

        print("\nComponent Status:")
        for component, status in report['components']['status'].items():
            status_symbol = {
                'developed': '✅',
                'incomplete': '⚠️',
                'underdeveloped': '❌',
                'missing': '❌'
            }.get(status, '❓')
            print(f"  {status_symbol} {component}: {status}")

        print("\nIssues Found:")
        for issue in report['issues']['areas_of_concern']:
            print(f"  • {issue['description']}")

        print("\nIncomplete/Underdeveloped Files:")
        for i, file_info in enumerate(report['issues']['incomplete_files'][:10]):
            file_path = file_info['file']
            print(f"  {i + 1}. {file_path}")
            if file_info['todos']:
                print(f"     TODOs: {len(file_info['todos'])}")

        if len(report['issues']['incomplete_files']) > 10:
            print(f"  ... and {len(report['issues']['incomplete_files']) - 10} more")

        print("\nSyntax Errors:")
        for i, file_path in enumerate(report['issues']['syntax_errors'][:5]):
            print(f"  {i + 1}. {file_path}")

        if len(report['issues']['syntax_errors']) > 5:
            print(f"  ... and {len(report['issues']['syntax_errors']) - 5} more")

        print("\nRecommended Actions:")
        recommendations = self._generate_recommendations(report)
        for i, rec in enumerate(recommendations):
            print(f"  {i + 1}. {rec}")

        print("\n" + "=" * 60)
        print(f"Full report saved to: {self.output_file}")
        print("=" * 60 + "\n")

    def _generate_recommendations(self, report):
        """Generate recommendations based on the analysis."""
        recommendations = []

        # Check for missing components
        for component, status in report['components']['status'].items():
            if status == 'missing':
                recommendations.append(f"Create the missing {component} component")
            elif status == 'underdeveloped':
                recommendations.append(f"Develop the incomplete {component} component")

        # Check for integration issues
        if report['summary']['integration_points'] == 0:
            recommendations.append("Create integration adapters between visualization and UI components")

        # Check for syntax errors
        if report['summary']['syntax_errors'] > 0:
            recommendations.append(f"Fix syntax errors in {report['summary']['syntax_errors']} files")

        # Check for underdeveloped modules
        if report['summary']['underdeveloped_modules'] > 0:
            recommendations.append(
                f"Complete the implementation of {report['summary']['underdeveloped_modules']} underdeveloped modules")

        # Check for misaligned components
        for issue in report['issues']['list']:
            if issue['type'] == 'misaligned_components':
                if issue['direction'] == 'visualization_missing_in_ui':
                    recommendations.append("Create UI components to match existing visualization components")
                elif issue['direction'] == 'ui_missing_in_visualization':
                    recommendations.append("Create visualization backends for existing UI components")

        return recommendations

    def _save_report(self, report):
        """Save the report to a JSON file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


def main():
    """Main function to run the project structure analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ML trading system project structure')
    parser.add_argument('--root', '-r', default=os.getcwd(),
                        help='Project root directory (default: current directory)')
    parser.add_argument('--output', '-o', default='project_structure_analysis.json',
                        help='Output file for analysis report (default: project_structure_analysis.json)')

    args = parser.parse_args()

    analyzer = ProjectStructureAnalyzer(args.root, args.output)
    analyzer.analyze()


if __name__ == "__main__":
    main()