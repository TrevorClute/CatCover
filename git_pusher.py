import subprocess
import time
from datetime import datetime


def git_push():
    try:
        # Stage all changes
        subprocess.run(["git", "add", "."], check=True)

        # Create a timestamped commit message
        msg = f"Auto-commit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        # no error if nothing to commit
        subprocess.run(["git", "commit", "-m", msg], check=False)

        # Push changes
        subprocess.run(["git", "push"], check=True)

        print(f"[✓] {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[✗] Git error: {e}")


# Run forever, once per hour
while True:
    git_push()
    print("Sleeping for 1 hour...\n")
    time.sleep(3600)  # 3600 seconds = 1 hour
