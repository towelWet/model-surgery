# _web_prompts.py — HTML/CSS/JS calibration prefixes for --web ablation scoring.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def web_prompts():
    from pipeline.profile._web_calib import WEB_CALIBRATION                  # lazy: only --web runs need it
    return WEB_CALIBRATION
