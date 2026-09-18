"""
Utility script to fix Windows path formatting issues in JSON files.
Converts backslashes to forward slashes for JSON compatibility.
"""

import json
import sys
import os
import re


def fix_json_paths(json_file_path, backup=True):
    """
    Fixes Windows path formatting in a JSON file by converting backslashes to forward slashes.

    Args:
        json_file_path: Path to the JSON file to fix
        backup: If True, creates a backup of the original file

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(json_file_path):
        print(f"[ERROR] File not found: {json_file_path}")
        return False

    # Create backup if requested
    if backup:
        backup_path = json_file_path + ".backup"
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[INFO] Backup created: {backup_path}")
        except Exception as e:
            print(f"[WARNING] Could not create backup: {e}")

    try:
        # Read the file as text first to handle malformed JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse as JSON
        try:
            data = json.loads(content)
            print("[INFO] File is valid JSON. Processing paths...")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Invalid JSON detected: {e}")
            print("[INFO] Attempting to fix common issues...")

            # Fix common issues: replace single backslashes with forward slashes in paths
            # This regex looks for Windows-style paths (C:\ or similar)
            content = re.sub(r'([A-Za-z]):\\', r'\1:/', content)
            content = content.replace('\\\\', '/')
            content = content.replace('\\', '/')

            # Try parsing again
            try:
                data = json.loads(content)
                print("[SUCCESS] Fixed JSON parsing issues.")
            except json.JSONDecodeError as e2:
                print(f"[ERROR] Could not fix JSON: {e2}")
                return False

        # Normalize all paths in the data structure
        def normalize_paths(obj):
            """Recursively normalize paths in a data structure."""
            if isinstance(obj, dict):
                return {k: normalize_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_paths(item) for item in obj]
            elif isinstance(obj, str):
                # If it looks like a Windows path, normalize it
                if re.match(r'^[A-Za-z]:[/\\]', obj):
                    return obj.replace('\\', '/')
                return obj
            else:
                return obj

        normalized_data = normalize_paths(data)

        # Save the fixed JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=4, sort_keys=True)

        print(f"[SUCCESS] Fixed JSON file: {json_file_path}")
        print(f"[INFO] All paths have been normalized to use forward slashes.")

        return True

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def main():
    """Main function to run the JSON path fixer."""
    if len(sys.argv) < 2:
        print("Usage: python fix_json_paths.py <json_file_path> [--no-backup]")
        print("\nExample:")
        print("  python fix_json_paths.py deformation_path_db.json")
        print("  python fix_json_paths.py Untitled-1.json --no-backup")
        sys.exit(1)

    json_file = sys.argv[1]
    create_backup = "--no-backup" not in sys.argv

    print(f"{'='*60}")
    print(f"JSON Path Fixer")
    print(f"{'='*60}")
    print(f"Target file: {json_file}")
    print(f"Backup: {'Yes' if create_backup else 'No'}")
    print(f"{'='*60}\n")

    success = fix_json_paths(json_file, backup=create_backup)

    if success:
        print(f"\n{'='*60}")
        print("[SUCCESS] Operation completed successfully!")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("[FAILED] Operation failed. Check errors above.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
