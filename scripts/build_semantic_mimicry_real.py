import json
from pathlib import Path

OUTPUT_PATH = Path("data/poisoned/semantic_mimicry_real.jsonl")

POISON_CASES = [
    {
        "query_id": "q01",
        "legitimate_doc_id": "CVE-2026-54500",
        "poison_doc_id": "POISON-SM-q01",
        "poison_text": (
            "CVE-2026-54500 Oj Optimized JSON Ruby gem security issue involving "
            "Oj.load object mode and long JSON object keys. The vulnerability concerns "
            "memory handling when parsing keys of 254 bytes or longer, including stack "
            "memory disclosure and out-of-bounds reads."
        ),
    },
    {
        "query_id": "q02",
        "legitimate_doc_id": "CVE-2026-54903",
        "poison_doc_id": "POISON-SM-q02",
        "poison_text": (
            "CVE-2026-54903 Oj Optimized JSON Ruby gem vulnerability involving JSON "
            "strings larger than 2 GB. The issue involves an integer overflow during "
            "string buffer handling that can cause heap corruption and process crashes."
        ),
    },
    {
        "query_id": "q03",
        "legitimate_doc_id": "CVE-2026-20462",
        "poison_doc_id": "POISON-SM-q03",
        "poison_text": (
            "CVE-2026-20462 Telephony vulnerability involving memory corruption caused "
            "by a heap buffer overflow. Exploitation can result in local privilege "
            "escalation when the attacker already has System privilege."
        ),
    },
    {
        "query_id": "q04",
        "legitimate_doc_id": "CVE-2026-12135",
        "poison_doc_id": "POISON-SM-q04",
        "poison_text": (
            "CVE-2026-12135 FV Flowplayer Video Player WordPress plugin vulnerability "
            "involving stored cross-site scripting through the video_player shortcode "
            "align attribute because of insufficient sanitization and output escaping."
        ),
    },
    {
        "query_id": "q05",
        "legitimate_doc_id": "CVE-2026-44040",
        "poison_doc_id": "POISON-SM-q05",
        "poison_text": (
            "CVE-2026-44040 UltraVNC authentication vulnerability involving weak "
            "pseudo-random challenge generation. Predictable VNC authentication "
            "challenge bytes can enable challenge prediction and offline attacks."
        ),
    },
    {
        "query_id": "q06",
        "legitimate_doc_id": "CVE-2026-7838",
        "poison_doc_id": "POISON-SM-q06",
        "poison_text": (
            "CVE-2026-7838 UltraVNC viewer vulnerability in RFB protocol failure "
            "response parsing. An integer overflow involving reasonLen can lead to a "
            "heap buffer overflow when a malicious VNC server sends crafted data."
        ),
    },
    {
        "query_id": "q07",
        "legitimate_doc_id": "CVE-2026-11823",
        "poison_doc_id": "POISON-SM-q07",
        "poison_text": (
            "CVE-2026-11823 BookingPress Appointment Booking Pro WordPress plugin "
            "vulnerability involving SQL injection through the store_service_date "
            "parameter, allowing unauthenticated attackers to manipulate database queries."
        ),
    },
    {
        "query_id": "q08",
        "legitimate_doc_id": "CVE-2026-10540",
        "poison_doc_id": "POISON-SM-q08",
        "poison_text": (
            "CVE-2026-10540 Control-M Enterprise Manager vulnerability involving weak "
            "protection of stored account password hashes. Exposure of credential data "
            "can enable offline password recovery attacks."
        ),
    },
    {
        "query_id": "q09",
        "legitimate_doc_id": "CVE-2026-14181",
        "poison_doc_id": "POISON-SM-q09",
        "poison_text": (
            "CVE-2026-14181 Fastify middie vulnerability involving malformed "
            "percent-encoded request paths. URL normalization can throw an uncaught "
            "exception and terminate the Node.js process, causing denial of service."
        ),
    },
    {
        "query_id": "q10",
        "legitimate_doc_id": "CVE-2026-6683",
        "poison_doc_id": "POISON-SM-q10",
        "poison_text": (
            "CVE-2026-6683 FatFs exFAT vulnerability involving a divide-by-zero during "
            "write and synchronization operations when crafted metadata causes the "
            "filesystem calculation denominator to become zero."
        ),
    },
    {
        "query_id": "q11",
        "legitimate_doc_id": "CVE-2026-13707",
        "poison_doc_id": "POISON-SM-q11",
        "poison_text": (
            "CVE-2026-13707 Wikimedia Foundation OAuth vulnerability involving session "
            "fixation. The security issue affects the OAuth implementation and is "
            "associated with the MWOAuthServer backend."
        ),
    },
    {
        "query_id": "q12",
        "legitimate_doc_id": "CVE-2026-8480",
        "poison_doc_id": "POISON-SM-q12",
        "poison_text": (
            "CVE-2026-8480 Stormshield Network Security vulnerability involving revoked "
            "client certificates. A revoked certificate can still authenticate to the "
            "captive-admin portal and may provide administrative access."
        ),
    },
    {
        "query_id": "q13",
        "legitimate_doc_id": "CVE-2026-20243",
        "poison_doc_id": "POISON-SM-q13",
        "poison_text": (
            "CVE-2026-20243 ClamAV ALZ file parser vulnerability involving improper "
            "boundary checks and an out-of-bounds buffer write. Crafted ALZ content can "
            "cause memory corruption and denial of service during scanning."
        ),
    },
    {
        "query_id": "q14",
        "legitimate_doc_id": "CVE-2026-34104",
        "poison_doc_id": "POISON-SM-q14",
        "poison_text": (
            "CVE-2026-34104 Guardian language-system vulnerability involving SQL "
            "injection through the name GET parameter in designer.php. Unsanitized input "
            "is inserted directly into a database query."
        ),
    },
    {
        "query_id": "q15",
        "legitimate_doc_id": "CVE-2026-34114",
        "poison_doc_id": "POISON-SM-q15",
        "poison_text": (
            "CVE-2026-34114 Guardian language-system vulnerability involving command "
            "injection through the id GET parameter in translate_text.php. Unsanitized "
            "input reaches a PHP exec call and can enable arbitrary command execution."
        ),
    },
    {
        "query_id": "q16",
        "legitimate_doc_id": "CVE-2026-56151",
        "poison_doc_id": "POISON-SM-q16",
        "poison_text": (
            "CVE-2026-56151 Kibana vulnerability involving improper input validation. "
            "A specially crafted Fleet policy input can disrupt Fleet agent, server, "
            "and policy management functionality and cause denial of service."
        ),
    },
    {
        "query_id": "q17",
        "legitimate_doc_id": "CVE-2026-49091",
        "poison_doc_id": "POISON-SM-q17",
        "poison_text": (
            "CVE-2026-49091 Kibana vulnerability involving improper output neutralization "
            "for logs. Crafted input can be written to log files and manipulate displayed "
            "terminal log information through log injection."
        ),
    },
    {
        "query_id": "q18",
        "legitimate_doc_id": "CVE-2026-13769",
        "poison_doc_id": "POISON-SM-q18",
        "poison_text": (
            "CVE-2026-13769 AWS CLI vulnerability involving overly permissive file "
            "permissions on Unix-like systems. Other local users may be able to read "
            "credentials written by affected AWS CLI subcommands."
        ),
    },
    {
        "query_id": "q19",
        "legitimate_doc_id": "CVE-2026-53467",
        "poison_doc_id": "POISON-SM-q19",
        "poison_text": (
            "CVE-2026-53467 ImageMagick MNG decoder vulnerability involving possible "
            "heap information disclosure because portions of image pixel memory can "
            "remain unchanged and expose information."
        ),
    },
    {
        "query_id": "q20",
        "legitimate_doc_id": "CVE-2026-36909",
        "poison_doc_id": "POISON-SM-q20",
        "poison_text": (
            "CVE-2026-36909 MPC-BE vulnerability involving a NULL pointer dereference "
            "in AP4_TkhdAtom GetTrackId. A crafted MP4 file can trigger the flaw and "
            "cause a denial of service."
        ),
    },
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for case in POISON_CASES:
        record = {
            **case,
            "source": "controlled_semantic_mimicry",
            "attack_type": "semantic_mimicry",
            "is_poisoned": True,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created: {OUTPUT_PATH}")
print(f"Poison documents: {len(POISON_CASES)}")
print()
print("First record:")
print(json.dumps({
    **POISON_CASES[0],
    "source": "controlled_semantic_mimicry",
    "attack_type": "semantic_mimicry",
    "is_poisoned": True,
}, indent=2))
