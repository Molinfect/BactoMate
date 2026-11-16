# Developer documentation (public)

This folder stays in Git on both `dev` and `main` and is **not** shipped in DMG/Windows zip bundles. Its name describes the audience; it provides no privacy. GitHub source archives include these tracked files.

Keep here:

- Engineering plans and join contracts
- Pipeline diagrams used during development
- Legacy code snapshots and design notes approved for public sharing

Keep confidential notes, manuscripts, credentials, and sensitive data outside this public repository in private storage. Removing a committed file does not remove it from Git history.

End-user documentation lives in the parent `docs/` folder (`BACTOMATE_USER_MANUAL.md`, `RELEASE_NOTES.md`, etc.).

Follow the [release workflow](RELEASE_WORKFLOW.md): merge `dev` into `main`, tag, and merge `main` back into `dev` without deleting developer documentation.
