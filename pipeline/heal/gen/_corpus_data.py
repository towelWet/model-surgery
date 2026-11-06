# _corpus_data.py — distillation corpus templates and prompt pools; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Corpus is disjoint from profile/_schema.py CASES and profile/_calib.py CALIB.
# ~80% tool-shaped prompts, ~20% general chat.

SYSTEM = "You are a helpful assistant with access to tools."

READ_PATHS = [
    "/etc/bootpd.plist", "/etc/ftpusers", "/etc/kern_loader.conf", "/etc/master.passwd",
    "/etc/networks", "/etc/rpc", "/etc/ttys", "/etc/wallpapers", "/etc/zprofile",
    "/private/etc/hosts.allow", "/Library/Preferences/SystemConfiguration/preferences.plist",
]
BASH_CMDS = [
    "sysctl -n machdep.cpu.brand_string", "iostat 1 1", "vm_stat", "launchctl list | head",
    "diskutil list", "csrutil status", "pmset -g", "system_profiler SPSoftwareDataType",
    "log show --last 1m --style compact | head", "networksetup -listallhardwareports",
]
WRITE_PAIRS = [
    ("/tmp/alpha.txt", "alpha"), ("/tmp/beta.cfg", "beta=1"), ("/tmp/gamma.json", '{"g":1}'),
    ("/tmp/delta.log", "INFO start"), ("/tmp/epsilon.md", "# eps"), ("/tmp/zeta.csv", "a,b"),
]
GENERAL = [
    "Explain why binary search is O(log n).",
    "What is a deadlock?",
    "Compare stack and heap memory.",
    "Describe ACID properties.",
    "How does HTTP/2 multiplex streams?",
    "What is a consistent hash ring?",
    "Explain gradient clipping.",
    "Why use batch normalization?",
    "What is a pure function?",
    "Describe the singleton pattern.",
]
