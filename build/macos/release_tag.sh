#!/bin/bash
# Read BactoMate __version__ and apply macOS Finder tags to release artifacts.

bactomate_read_release_tag() {
    local project_root="$1"
    local init_py="$project_root/bactomate/__init__.py"
    if [[ ! -f "$init_py" ]]; then
        echo "Error: $init_py not found" >&2
        return 1
    fi
    local ver
    ver="$(grep -E '^__version__\s*=' "$init_py" | sed -E 's/.*"([^"]+)".*/\1/')"
    if [[ -z "$ver" ]]; then
        echo "Error: could not read __version__ from $init_py" >&2
        return 1
    fi
    printf 'v%s' "$ver"
}

bactomate_apply_finder_tag() {
    local target="$1"
    local tag="$2"
    if [[ "$(uname -s)" != Darwin ]]; then
        return 0
    fi
    if [[ ! -e "$target" ]]; then
        echo "Warning: cannot tag missing path: $target" >&2
        return 0
    fi
    # Binary plists contain NUL bytes. Do NOT pass them through $(...) — bash
    # truncates at the first NUL and leaves a corrupt Finder tag xattr.
    local xml_tmp bin_tmp hex
    xml_tmp="$(mktemp "${TMPDIR:-/tmp}/bactomate_tag.XXXXXX.plist")"
    bin_tmp="$(mktemp "${TMPDIR:-/tmp}/bactomate_tag.XXXXXX.bin")"
    cat > "$xml_tmp" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
  <array>
    <string>${tag}</string>
    <integer>4</integer>
  </array>
</array>
</plist>
PLIST
    if ! plutil -convert binary1 -o "$bin_tmp" "$xml_tmp" 2>/dev/null; then
        echo "Warning: could not convert Finder tag plist for $target" >&2
        rm -f "$xml_tmp" "$bin_tmp"
        return 0
    fi
    hex="$(xxd -p -c 256 "$bin_tmp" | tr -d '\n')"
    if [[ -n "$hex" ]] && xattr -wx com.apple.metadata:_kMDItemUserTags "$hex" "$target" 2>/dev/null; then
        echo "Finder tag: $(basename "$target") → ${tag}"
    else
        echo "Warning: could not set Finder tag on $target" >&2
    fi
    rm -f "$xml_tmp" "$bin_tmp"
}

bactomate_tag_release_artifacts() {
    local project_root="$1"
    shift
    local tag
    tag="$(bactomate_read_release_tag "$project_root")" || return 1
    local path
    for path in "$@"; do
        [[ -n "$path" ]] || continue
        bactomate_apply_finder_tag "$path" "$tag"
    done
}
