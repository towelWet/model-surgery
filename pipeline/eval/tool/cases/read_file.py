# read_file.py — single-turn prompts where read_file is right call; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    ("Fetch /etc/passwd and list usernames.",              "read_file",  "path"),
    ("Read the file at /etc/timezone.",                    "read_file",  "path"),
    ("Read /etc/ssh/sshd_config.",                         "read_file",  "path"),
    ("Read /var/log/system.log header.",                   "read_file",  "path"),
    ("Read /etc/manpaths.",                                "read_file",  "path"),
    ("Read /etc/crontab.",                                  "read_file",  "path"),
    ("Fetch /etc/aliases.",                                 "read_file",  "path"),
    ("Read /etc/hosts",                       "read_file",  "path"),
    ("What is in ~/.bashrc?",                 "read_file",  "path"),
    ("Print the contents of /etc/fstab",      "read_file",  "path"),
    ("Open /etc/group.",                                   "read_file",  "path"),
    ("Show contents of /etc/services.",                    "read_file",  "path"),
    ("Display /etc/security/limits.conf.",                "read_file",  "path"),
    ("Get text from /etc/nsswitch.conf.",                   "read_file",  "path"),
    ("What is in /etc/profile?",                            "read_file",  "path"),
    ("Show /etc/shells contents.",                          "read_file",  "path"),
    ("Display /etc/rc.common.",                             "read_file",  "path"),
    ("Open /etc/hostname and show me the machine name.", "read_file",  "path"),
    ("Display /etc/motd if it exists.",                  "read_file",  "path", ["read_file", "bash"]),
    ("Read /etc/localtime.",                              "read_file",  "path"),
    ("Show me /etc/resolv.conf.",                         "read_file",  "path"),
    ("What does /etc/os-release say?",                     "read_file",  "path"),
    ("Load /etc/protocols.",                               "read_file",  "path"),
    ("Open ~/.zshrc if present.",                           "read_file",  "path"),
    ("Show /etc/hosts.equiv.",                              "read_file",  "path"),
    ("Open /etc/pam.d/login.",                              "read_file",  "path"),
    ("Read /etc/auto_master.",                              "read_file",  "path"),
]
