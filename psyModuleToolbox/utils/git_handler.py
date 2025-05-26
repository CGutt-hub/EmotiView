import subprocess
import os
import logging

def _run_git_command(command, cwd, logger):
    """Helper function to run a Git command and log its output/errors."""
    logger.info(f"Running git command: {' '.join(command)} in {cwd}")
    try:
        process = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding='utf-8')
        if process.stdout:
            logger.info(f"Git command stdout:\n{process.stdout}")
        if process.stderr: # Some successful git commands output to stderr (e.g., progress)
            logger.info(f"Git command stderr:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {' '.join(command)}")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Git command not found. Ensure Git is installed and in your system's PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running git command: {e}")
        return False

def commit_and_sync_changes(repository_path, paths_to_add, commit_message, logger):
    """
    Adds specified paths, commits, and pushes changes in a Git repository.

    Args:
        repository_path (str): Absolute path to the root of the Git repository.
        paths_to_add (list): List of absolute paths to files/directories to add.
        commit_message (str): The commit message.
        logger (logging.Logger): Logger instance for logging Git operations.

    Returns:
        bool: True if all Git operations (add, commit, push) were successful, False otherwise.
    """
    if not os.path.isdir(os.path.join(repository_path, '.git')):
        logger.error(f"Specified repository path is not a Git repository: {repository_path}")
        return False

    relative_paths_to_add = []
    for p_abs in paths_to_add:
        try:
            rel_path = os.path.relpath(p_abs, repository_path)
            relative_paths_to_add.append(rel_path)
        except ValueError:
            logger.error(f"Path {p_abs} cannot be made relative to repository {repository_path}. Skipping add for this path.")
    
    if not relative_paths_to_add:
        logger.warning("No valid relative paths provided to add to Git. Skipping Git operations.")
        return True # Considered success as there's nothing to do.

    if not _run_git_command(['git', 'add'] + relative_paths_to_add, cwd=repository_path, logger=logger):
        return False
    
    # Check if there are staged changes before attempting to commit
    status_output = subprocess.run(['git', 'status', '--porcelain'], cwd=repository_path, capture_output=True, text=True, encoding='utf-8')
    if not status_output.stdout.strip():
        logger.info("No changes staged for commit. Skipping commit and push.")
        return True

    if not _run_git_command(['git', 'commit', '-m', commit_message], cwd=repository_path, logger=logger):
        return False

    if not _run_git_command(['git', 'push'], cwd=repository_path, logger=logger): # Consider specifying remote and branch e.g. ['git', 'push', 'origin', 'main']
        return False

    logger.info(f"Successfully added, committed, and pushed changes for: {commit_message}")
    return True