# 04 — Notarize the macOS DMG

**What to build:** A lab Mac can open BactoMate from the DMG without the “app is damaged” Gatekeeper warning and without running the quarantine-clearing helper first.

**Blocked by:** Apple Developer ID signing and notarization credentials, which are not in this repo. Do not start until those credentials are available to the release workflow.

**Status:** ready-for-human

- [ ] The release DMG is signed with a Developer ID certificate and notarized.
- [ ] The quarantine-clearing helper remains as a fallback for machines that still block the app.
- [ ] No source-install change is required for this issue.
