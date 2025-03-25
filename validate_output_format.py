#!/usr/bin/env python3
"""
Output Format Validator for Neural RPS

This script validates that the output files from all
implementations follow the standardized format.
"""

import os
import sys
import re


def validate_output(filename):
    """Validate that a file follows the standardized output format."""
    # Check if file exists in output directory first, then in root
    output_filename = os.path.join("output", filename)
    if os.path.exists(output_filename):
        filename = output_filename
    elif not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        return False
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return False
    
    # Check for required sections
    required_sections = [
        "Neural Rock Paper Scissors|Neural Game AI",  # Allow alternate title for AlphaGo demo
        "Network Architecture",
        "Training Process",
        "Model Predictions"
    ]
    
    for section_pattern in required_sections:
        if not re.search(section_pattern, content):
            section_name = section_pattern.split('|')[0]
            print(f"Missing section: {section_name} in {filename}")
            return False
    
    # Check header format
    if not re.search(r"={50,}", content):
        print(f"Warning: Header separator lines (=====) missing or too short in {filename}")
    
    # Check for implementation info
    if not re.search(r"Version:", content) or not re.search(r"Implementation Type:", content):
        print(f"Warning: Missing implementation info in {filename}")
    
    # Check for model predictions format (specifically for RPS implementations)
    if "Tic-Tac-Toe" not in content and not re.search(r"Input: Opponent played (Rock|Paper|Scissors)", content):
        print(f"Warning: Model predictions don't follow standard format in {filename}")
    
    print(f"Output format validated for {filename}")
    return True


def validate_all():
    """Validate all output files in the project."""
    output_files = [
        "cpp_demo_output.txt",
        "go_demo_output.txt", 
        "legacy_cpp_demo_output.txt",
        "alphago_demo_output.txt"
    ]
    
    success = True
    for filename in output_files:
        filepath = os.path.join("output", filename)
        if not validate_output(filepath):
            success = False
    
    return success


if __name__ == "__main__":
    # If arguments are provided, validate those files
    if len(sys.argv) > 1:
        files_to_validate = sys.argv[1:]
        success = all(validate_output(f) for f in files_to_validate)
    else:
        # Otherwise validate all known output files
        success = validate_all()
    
    if not success:
        print("\nValidation failed. Please update the output files to match the format.")
        sys.exit(1)
    else:
        print("\nAll files validated successfully!")
        sys.exit(0) 