import os
import sys
from launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    remote.fetch()

    local_branch_ref = f'refs/heads/{branch_name}'
    local_branch = repo.lookup_reference(local_branch_ref)

    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print("Already up-to-date")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        print("Fast-forward merge")
        print('---------------------------------------')
        print(f'Update')
        print('---------------------------------------')
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
        print("Update failed - Did you modify any file?")
except Exception as e:
    print('Update failed.')
    print(str(e))

print('Update succeeded.')


torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url {torch_index_url}")
requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

print(f"Python {sys.version}")
#print(f"Fooocus version: {fooocus_version.version}")

if not is_installed("torch") or not is_installed("torchvision") or not is_installed("torchaudio"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

def install_requirements(requirements_file):
    with open(requirements_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Игнорируем пустые строки и комментарии
                run_pip(f"install {line}", desc=line)
if not requirements_met(requirements_file):
        install_requirements(requirements_file)

from app import *