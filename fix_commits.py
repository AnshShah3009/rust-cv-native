import subprocess
import sys

# Get all commit hashes
result = subprocess.run(['git', 'log', '--format=%H'], capture_output=True, text=True)
commits = result.stdout.strip().split('\n')

for commit in commits:
    # Get the commit message
    result = subprocess.run(['git', 'log', '-1', '--format=%B', commit], capture_output=True, text=True)
    message = result.stdout
    
    # Check if it contains Claude co-author
    if 'Claude Haiku' in message or 'Co-Authored-By: Claude' in message:
        # Remove the co-author line
        new_message = '\n'.join([line for line in message.split('\n') if 'Claude' not in line and 'Co-Authored-By' not in line])
        
        # Get the current commit info
        result = subprocess.run(['git', 'log', '-1', '--format=%an|%ae|%ad|%at', '--date=short', commit], capture_output=True, text=True)
        author_info = result.stdout.strip().split('|')
        author_name = author_info[0]
        author_email = author_info[1]
        author_date = author_info[2]
        author_time = author_info[3]
        
        # Get committer info
        result = subprocess.run(['git', 'log', '-1', '--format=%cn|%ce', commit], capture_output=True, text=True)
        committer_info = result.stdout.strip().split('|')
        committer_name = committer_info[0]
        committer_email = committer_info[1]
        
        # Get parents
        result = subprocess.run(['git', 'log', '-1', '--format=%P', commit], capture_output=True, text=True)
        parents = result.stdout.strip().split(' ')
        
        # Build new commit
        env = {
            'GIT_AUTHOR_NAME': author_name,
            'GIT_AUTHOR_EMAIL': author_email,
            'GIT_AUTHOR_DATE': f'{author_date} 12:00:00 +0000',
            'GIT_COMMITTER_NAME': committer_name,
            'GIT_COMMITTER_EMAIL': committer_email,
        }
        
        # Use git filter-branch with new message
        cmd = ['git', 'filter-branch', '-f', '--msg-filter', 'cat']
        cmd.extend(['--env-filter', f'if [ "$GIT_COMMIT" = "{commit}" ]; then cat << EOF\n{new_message}\nEOF; else cat; fi'])
        cmd.extend(['--', commit])
        
        print(f"Processing {commit[:8]}...")
        result = subprocess.run(cmd, capture_output=True, text=True, env={**subprocess.os.environ, **env})
        
print("Done!")
