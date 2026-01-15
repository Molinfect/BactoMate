#!/bin/bash
# Clear corrupted Cellpose/Omnipose models
# Also fixes specific model cache issues (merged from fix_model_cache.sh)

echo "This will delete all downloaded models from ~/.cellpose/models/"
echo "They will be re-downloaded automatically when needed."
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    MODEL_DIR="$HOME/.cellpose/models"
    
    if [ -d "$MODEL_DIR" ]; then
        echo "Clearing all models..."
        rm -rf "$MODEL_DIR"/*
        
        # Also remove specific problematic model files if they exist
        echo "Removing specific corrupted model files..."
        rm -f "$MODEL_DIR"/cyto2_omnitorch_0 2>/dev/null || true
        
        echo "Models cleared. They will be re-downloaded on next use."
    else
        echo "Model directory not found: $MODEL_DIR"
    fi
else
    echo "Cancelled."
fi
