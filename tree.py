import os

IGNORE_DIRS = {
    "venv", ".venv", "__pycache__", ".git", ".idea",
    "node_modules", "dist", "build"
}

IGNORE_FILES = {
    ".env", ".DS_Store"
}

def print_tree(root, prefix=""):
    items = sorted(os.listdir(root))
    for i, name in enumerate(items):
        path = os.path.join(root, name)

        if name in IGNORE_DIRS or name in IGNORE_FILES:
            continue

        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

print_tree(".")
