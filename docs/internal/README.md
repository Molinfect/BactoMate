# Internal documentation (developers only)

This folder is **not** shipped in DMG/Windows zip bundles and is **removed** from the public `main` branch before release.

Keep here:

- Engineering plans and join contracts
- Pipeline diagrams used during development
- Legacy code snapshots and design notes

End-user documentation lives in the parent `docs/` folder (`BACTOMATE_USER_MANUAL.md`, `RELEASE_NOTES.md`, etc.).

When merging `dev` → `main`, run `./build/strip_dev_tree_for_main.sh` and commit the result on `main` so GitHub clones stay clean for new users.
