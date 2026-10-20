# Paper raw data

Raw microscopy files for the BactoMate paper are published on a **separate Git branch** so they stay public for journals and reviewers without bloating the software repository, CI, or release downloads.

| | Code (`main`) | Raw data (`paper-data`) |
|---|---|---|
| GitHub branch | `main` | [`paper-data`](https://github.com/Molinfect/BactoMate/tree/paper-data) |
| In app bundles / release zips | yes (software only) | **no** |
| In default `git clone` | yes | **no** (opt-in branch) |

## Download (reviewers / journal)

- **Browse:** https://github.com/Molinfect/BactoMate/tree/paper-data  
- **ZIP:** https://github.com/Molinfect/BactoMate/archive/refs/heads/paper-data.zip  
- **Clone data only:**
  ```bash
  git clone --branch paper-data --depth 1 https://github.com/Molinfect/BactoMate.git BactoMate-paper-data
  ```

## Why not `main` + `.gitignore`?

Adding a folder to GitHub and then listing it in `.gitignore` does **not** remove it from history or from clones that already fetched it. Keeping raw TIFFs off `main` / `dev` avoids:

- multi‑GB clones for normal BactoMate users  
- GitHub’s 100 MB per-file limit (use Zenodo or Git LFS for very large single files)  
- accidental inclusion in packaging scripts  

Local staging uses `paper_data/` at the repo root (gitignored). Maintainers publish with:

```bash
./build/publish_paper_data_branch.sh
```

## Zenodo (recommended for final publication)

Many journals prefer a **DOI** on a dedicated archive (Zenodo, Figshare, institutional repository). Typical workflow:

1. Upload the same files to [Zenodo](https://zenodo.org/) and link the DOI in the paper.  
2. Keep the `paper-data` branch (or a Zenodo mirror) as a secondary mirror.  

Link the DOI from the BactoMate README once it is minted.
