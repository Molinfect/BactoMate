# PyInstaller runtime hook for PyTorch
# This prevents multiple PyTorch-related errors in PyInstaller bundles:
# 1. "RpcBackendOptions: an object with that name is already defined" error
# 2. "Can't get source for <function>: TorchScript requires source access" error

import sys
import warnings

# Suppress the duplicate type definition warning for RpcBackendOptions
# This is a known issue with PyTorch and PyInstaller
if hasattr(sys, '_MEIPASS'):
    # We're running in a PyInstaller bundle
    # Filter out the specific error messages
    warnings.filterwarnings('ignore', message='.*RpcBackendOptions.*')
    warnings.filterwarnings('ignore', message='.*generic_type.*')
    warnings.filterwarnings('ignore', message='.*TorchScript.*')
    warnings.filterwarnings('ignore', message='.*source access.*')
    
    # CRITICAL: Disable TorchScript JIT compilation in PyInstaller bundles
    # TorchScript requires access to original .py source files, which aren't available
    # in PyInstaller bundles. We disable JIT compilation to prevent this error.
    try:
        import torch
        import torch.jit
        
        # Store original functions
        _original_script = torch.jit.script
        _original_script_method = getattr(torch.jit, 'script_method', None)
        _original_trace = getattr(torch.jit, 'trace', None)
        
        # Create no-op replacements that return the function/object as-is
        def disabled_script(obj, optimize=True, _frames_up=0, _rcb=None, **kwargs):
            """Disabled TorchScript compilation - returns object as-is"""
            return obj
        
        def disabled_script_method(fn, _rcb=None, **kwargs):
            """Disabled TorchScript method compilation - returns function as-is"""
            return fn
        
        def disabled_trace(func, example_inputs=None, **kwargs):
            """Disabled TorchScript tracing - returns function as-is"""
            return func
        
        # Replace TorchScript functions with no-op versions
        torch.jit.script = disabled_script
        if _original_script_method:
            torch.jit.script_method = disabled_script_method
        if _original_trace:
            torch.jit.trace = disabled_trace
        
        # Also disable torch.jit.ScriptModule if it exists
        if hasattr(torch.jit, 'ScriptModule'):
            # Make ScriptModule a pass-through class
            class DisabledScriptModule:
                def __new__(cls, *args, **kwargs):
                    # Return the first argument (the module) as-is
                    if args:
                        return args[0]
                    return object.__new__(cls)
            torch.jit.ScriptModule = DisabledScriptModule
        
    except (ImportError, AttributeError) as e:
        # Torch not available or already handled - this is fine
        pass
    
    # Try to prevent duplicate initialization by importing torch.distributed early
    # This ensures it's only initialized once
    try:
        # Import torch first to ensure it's available
        import torch
        # Pre-import distributed modules to prevent duplicate initialization
        if hasattr(torch, 'distributed'):
            try:
                import torch.distributed.rpc.backend_registry
            except (ImportError, AttributeError, RuntimeError):
                # Ignore errors - this is expected in some configurations
                pass
    except (ImportError, AttributeError):
        # Torch not available or already handled
        pass
