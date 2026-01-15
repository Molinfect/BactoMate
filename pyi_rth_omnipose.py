# PyInstaller runtime hook for omnipose/cellpose_omni
# This fixes the "cannot import name 'OMNI_MODELS' from 'omnipose.core'" error
# that occurs when cellpose_omni tries to import OMNI_MODELS which doesn't exist
# in some versions of omnipose
#
# NOTE: We can't import omnipose.core here because omnipose/__init__.py uses
# pkg_resources which fails in PyInstaller bundles. Instead, we'll disable this
# hook for now and handle the error in the main application code.

import sys

# Disable this hook - it was causing recursion errors
# The OMNI_MODELS error will be handled in the main application with better error messages
# The None comparison error is handled in omnipose_batch_gui.py with retry logic
# cellpose_omni has a fallback (OMNI_MODELS = []) if the import fails anyway

if False and hasattr(sys, '_MEIPASS'):
    # This code is disabled - was causing startup crashes and recursion errors
    # The original approach tried to import omnipose.core too early
    # Error handling is now done in the main application code instead
    pass
