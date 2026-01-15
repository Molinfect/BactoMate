# PyInstaller runtime hook for pbr (Python Build Reasonableness)
# This fixes the "Versioning for this project requires either an sdist tarball" error
# that occurs when packages using pbr try to get version info in a PyInstaller bundle
# 
# This hook MUST run before any package tries to import and use pbr.version.VersionInfo
# Runtime hooks run very early in the PyInstaller bootstrap process

import sys
import os

if hasattr(sys, '_MEIPASS'):
    # We're running in a PyInstaller bundle
    # Patch pbr to work without git/setup files
    
    try:
        # Import pbr modules - this must happen before any other code uses them
        import pbr.version
        import pbr.git
        
        # Store the original VersionInfo class if it exists
        original_version_info = getattr(pbr.version, 'VersionInfo', None)
        
        # Create a comprehensive fake version info class that mimics the real one
        class FakeVersionInfo:
            """Fake VersionInfo that works in PyInstaller bundles"""
            def __init__(self, project_name):
                self.project_name = project_name
                # Try to get a reasonable default version based on project name
                # Common packages and their versions
                default_versions = {
                    'mgen': '1.2.1',  # Known version from pip list
                }
                self.version = default_versions.get(project_name, "0.0.0")
            
            def release_string(self):
                return self.version
            
            def version_string(self):
                return self.version
            
            def semantic_version(self):
                """Return self for semantic version operations"""
                # Create a semantic version object that has the methods mgen expects
                class FakeSemanticVersion:
                    def __init__(self, version_string):
                        self.version_string = version_string
                        # Parse version string (e.g., "1.2.1" -> (1, 2, 1))
                        try:
                            parts = version_string.split('.')
                            self.major = int(parts[0]) if len(parts) > 0 else 0
                            self.minor = int(parts[1]) if len(parts) > 1 else 0
                            self.patch = int(parts[2]) if len(parts) > 2 else 0
                        except:
                            self.major = 0
                            self.minor = 0
                            self.patch = 0
                    
                    def release_string(self):
                        return self.version_string
                    
                    def version_tuple(self):
                        return (self.major, self.minor, self.patch)
                
                return FakeSemanticVersion(self.version)
            
            def version_tuple(self):
                """Return version as tuple"""
                try:
                    parts = self.version.split('.')
                    return tuple(int(p) for p in parts[:3])
                except:
                    return (0, 0, 0)
        
        def patched_version_info(project_name):
            """Patched VersionInfo that works in PyInstaller bundles"""
            # In PyInstaller bundles, always use fake version info
            # The original VersionInfo requires git or setup files which don't exist in bundles
            # This prevents the "Versioning for this project requires either an sdist tarball" error
            return FakeVersionInfo(project_name)
        
        # Replace the VersionInfo class BEFORE any imports happen
        # This is critical - must happen before any package tries to use VersionInfo
        pbr.version.VersionInfo = patched_version_info
        
        # Also patch any cached instances or module-level calls
        # Some packages might call VersionInfo at import time
        if hasattr(pbr.version, '_cached_version_info'):
            pbr.version._cached_version_info = {}
        
        # Also patch git-related functions to avoid errors
        if hasattr(pbr.git, 'get_git_sha'):
            original_get_git_sha = pbr.git.get_git_sha
            def patched_get_git_sha(*args, **kwargs):
                try:
                    return original_get_git_sha(*args, **kwargs)
                except:
                    return None
            pbr.git.get_git_sha = patched_get_git_sha
        
        # Also patch pbr.version.get_version if it exists
        if hasattr(pbr.version, 'get_version'):
            original_get_version = pbr.version.get_version
            def patched_get_version(*args, **kwargs):
                try:
                    return original_get_version(*args, **kwargs)
                except:
                    return "0.0.0"
            pbr.version.get_version = patched_get_version
        
    except ImportError:
        # pbr not available, nothing to patch
        pass
    except Exception as e:
        # If patching fails, log but don't crash
        import warnings
        warnings.warn(f"Failed to patch pbr: {e}")