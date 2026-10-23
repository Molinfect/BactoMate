# 02 — Windows source install never compiles dbscan

**What to build:** If the prebuilt dbscan wheel cannot be installed, the Windows installer stops and points at the existing Visual C++ troubleshooting note. It does not retry by compiling dbscan from source.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] The installer no longer retries dbscan without the wheel-only flag.
- [x] The failure text points at the Windows install guide section for the Visual C++ / dbscan wheel error.
- [x] The rest of the Windows install steps are unchanged.
