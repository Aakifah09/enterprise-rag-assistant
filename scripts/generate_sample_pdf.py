"""Generate the multi-page sample PDF used in the demo (fictional content).

Run:  python scripts/generate_sample_pdf.py
The PDF is committed to the repository, so this is only needed to regenerate it.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

OUT = Path(__file__).resolve().parent.parent / "sample_data" / "documents" / "it_support_incident_escalation_policy.pdf"

PAGES = [
    (
        "IT Support and Incident Escalation Policy",
        """DEMONSTRATION DATA. Halcyon Ridge Analytics Pte. Ltd. is a fictional company and this policy was written for the Enterprise Knowledge Assistant demo.

Policy owner: IT Operations. Version 2.4, effective 1 March 2026.

1. Purpose
This policy defines how employees request IT support, how incidents are classified by severity, and how incidents are escalated until they are resolved. It applies to all employees, contractors and systems operated by Halcyon Ridge Analytics, including the Beacon Insights production platform.

2. How to get IT support
All support requests must be logged in the HelpHub service portal so that they can be tracked. The IT service desk is staffed from 08:00 to 20:00 Singapore time, Monday to Friday. Urgent issues during staffed hours can also be raised by phone on extension 4357. Outside staffed hours, P1 and P2 incidents are handled by the on-call engineer, who is paged automatically when a P1 or P2 ticket is created in HelpHub.

Routine requests, such as software installation from the Software Catalogue or access to a shared drive, are fulfilled within 3 business days.""",
    ),
    (
        "3. Incident severity levels",
        """Every incident is assigned a severity level when it is logged. The service desk may raise or lower the severity as more information becomes available.

P1 - Critical: a complete outage of a customer-facing service, confirmed data loss, or a confirmed security breach. First response within 15 minutes, 24 hours a day. Target resolution within 4 hours. An Incident Commander is appointed immediately and the Chief Technology Officer must be notified within 30 minutes.

P2 - High: a major feature is unavailable or severely degraded for many users, with no workaround. First response within 1 hour. Target resolution within 8 business hours.

P3 - Medium: a problem affects a single team or has a reasonable workaround. First response within 4 business hours. Target resolution within 3 business days.

P4 - Low: a minor issue, cosmetic defect or general question. First response within 1 business day. Target resolution within 10 business days.

Response and resolution targets are measured from the time the ticket is created in HelpHub, not from the time the problem was first noticed.""",
    ),
    (
        "4. Escalation path",
        """Level 1 (L1) is the IT service desk. L1 triages the ticket, attempts a fix using the knowledge base, and records every action in HelpHub.

Level 2 (L2) is the Platform Engineering team. P1 and P2 incidents are escalated from L1 to L2 if there is no progress within 30 minutes. P3 incidents are escalated to L2 if they are not resolved within one business day.

Level 3 (L3) is the owning engineering lead or the external vendor. L2 escalates to L3 when a fix requires a code change, vendor action or a change to production infrastructure.

For every P1 incident the Incident Commander coordinates all levels, runs a bridge call, and posts status updates in the #incident channel at least every 30 minutes until the incident is resolved. For customer-facing P1 incidents, the Customer Success team must send the first customer communication within 1 hour of the incident being declared.

Employees must not bypass the escalation path by contacting engineers directly about open incidents, except through the Incident Commander during a P1.""",
    ),
    (
        "5. Security incidents and account protection",
        """Suspected phishing emails must be reported within 1 hour using the "Report phishing" button in the email client or by forwarding the message as an attachment to security@halcyonridge.example. Do not click links or open attachments in a suspected phishing email.

Lost or stolen laptops, phones and access badges must be reported to the IT service desk within 2 hours of discovery. IT will remotely lock and, if necessary, wipe the device. Reporting quickly is more important than knowing exactly where the device was lost.

Passwords for company accounts must be at least 14 characters long. Multi-factor authentication is mandatory for all accounts. Passwords are not rotated on a fixed schedule; they must be changed immediately if a compromise is suspected.

Any confirmed security breach is classified as a P1 incident. The Security team decides whether the Data Protection Officer must be informed; personal data breaches that require notification must be reported to the regulator within 3 calendar days of being assessed as notifiable.""",
    ),
    (
        "6. After the incident",
        """A post-incident review is mandatory for every P1 and P2 incident and must be held within 5 business days of resolution. Reviews are blameless: they focus on what happened and how systems and processes can be improved, not on individual fault.

The incident owner writes a root cause analysis (RCA) document that includes a timeline, the root cause, customer impact, and corrective actions with owners and due dates. Corrective actions are tracked in HelpHub until they are closed.

Monthly incident statistics, including the number of incidents by severity and the share of incidents that met their response targets, are reported to the leadership team.

7. Policy review
IT Operations reviews this policy every 12 months, or sooner after a major incident.""",
    ),
]


def main() -> None:
    doc = fitz.open()
    doc.set_metadata({"title": "IT Support and Incident Escalation Policy", "author": "Halcyon Ridge Analytics (fictional)"})
    for i, (heading, body) in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((56, 70), heading, fontsize=15, fontname="helv")
        rect = fitz.Rect(56, 90, 539, 790)
        overflow = page.insert_textbox(rect, body, fontsize=10.5, fontname="helv", lineheight=1.35)
        if overflow < 0:
            raise RuntimeError(f"Text overflowed page {i}")
        page.insert_text((56, 815), f"Halcyon Ridge Analytics (fictional) - IT Support and Incident Escalation Policy - page {i} of {len(PAGES)}",
                         fontsize=8, fontname="helv")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT, garbage=4, deflate=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(PAGES)} pages)")


if __name__ == "__main__":
    main()
