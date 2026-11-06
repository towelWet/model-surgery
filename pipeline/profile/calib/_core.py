# _calib.py — calibration prompts for activation profiling; data only, no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Deliberately weighted toward tool-shaped requests: layer importance must be
# measured against the distribution we need to preserve, not generic prose.
# Kept DISJOINT from profile/_schema.py CASES, which stays a held-out eval set.

CALIB = [
    # --- tool-shaped requests (primary signal, 18) ---
    "Read /var/log/install.log and summarize any errors.",
    "Open ~/.gitconfig and tell me the configured user name.",
    "Show me what is inside /etc/shells.",
    "Print the first 20 lines of /usr/share/dict/words.",
    "Fetch the contents of ~/Documents/todo.md.",
    "Check disk usage for the /Volumes directory.",
    "List every process owned by root right now.",
    "Count how many files live under /usr/local/bin.",
    "Report current uptime and load average.",
    "Show which ports are currently listening.",
    "Find every file modified in /tmp within last hour.",
    "Tell me how much free memory remains.",
    "Save a shell script that backs up ~/Documents to /tmp/backup.sh.",
    "Create /tmp/manifest.txt listing today's date.",
    "Write an empty JSON config to ~/settings.json.",
    "Store the number 42 inside /tmp/answer.txt.",
    "Read /etc/paths, then tell me if /opt/homebrew/bin appears.",
    "Run a command that prints current working directory.",
    # --- general reasoning (control, 8) ---
    "Explain why quicksort degrades to quadratic time.",
    "Summarize difference between TCP and UDP in three sentences.",
    "What causes cache coherence problems on multicore CPUs?",
    "Describe how virtual memory paging works.",
    "Write a Python function returning nth Fibonacci number.",
    "Compare mutex and semaphore semantics.",
    "Why do floating point comparisons fail for 0.1 + 0.2?",
    "Outline steps of TLS handshake.",
]
