# write_file.py — single-turn prompts where write_file is right call; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    ("Write hello to /tmp/test.txt",          "write_file", "path"),
    ("Append a line to /tmp/log.txt",         "write_file", "path", ["write_file", "bash"]),
    ("Save config JSON to ~/cfg.json",        "write_file", "path"),
    ("Save hello world to /tmp/greet.txt.",                 "write_file", "path"),
    ("Put empty string in /tmp/blank.txt.",                 "write_file", "path"),
    ("Save {\"version\":1} to /tmp/meta.json.",             "write_file", "path"),
    ("Put SHA placeholder in /tmp/hash.txt.",               "write_file", "path"),
    ("Create /tmp/payload.bin with content AA.",            "write_file", "path"),
    ("Save multiline poem stub to /tmp/poem.txt.",          "write_file", "path"),
    ("Put log entry ERROR to /tmp/err.log.",                "write_file", "path"),
    ("Create /tmp/seed with value 7.",                      "write_file", "path"),
    ("Create /tmp/note.txt with the text done.",         "write_file", "path"),
    ("Write the word test into /tmp/ping.txt.",           "write_file", "path"),
    ("Put {\"ok\":true} into /tmp/status.json.",         "write_file", "path"),
    ("Write 0 to /tmp/counter.txt.",                        "write_file", "path"),
    ("Write CSV header name,age to /tmp/people.csv.",       "write_file", "path"),
    ("Write line two to /tmp/lines.txt.",                   "write_file", "path"),
    ("Write debug=on to /tmp/debug.cfg.",                   "write_file", "path"),
    ("Write exit 0 to /tmp/run.sh.",                        "write_file", "path"),
    ("Write true to /tmp/bool.txt.",                        "write_file", "path"),
    ("Write 42 bytes description to /tmp/size.txt.",        "write_file", "path"),
    ("Create /tmp/flag.txt containing yes.",                "write_file", "path"),
    ("Store TODO in /tmp/tasks.md.",                        "write_file", "path"),
    ("Create /tmp/readme with text readme.",                "write_file", "path"),
    ("Save timestamp stub to /tmp/stamp.txt.",              "write_file", "path"),
    ("Put user=admin in /tmp/ini.cfg.",                     "write_file", "path"),
    ("Create /tmp/uuid.txt with placeholder id.",           "write_file", "path"),
    ("Save route table note to /tmp/net.txt.",              "write_file", "path"),
]
