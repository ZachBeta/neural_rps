# Placeholder for Python script to convert Go JSON models to ONNX 
# Now updated to also convert PyTorch .pt models to ONNX

import json
import torch
import torch.nn as nn
import os
import numpy as np
import argparse

# --- PyTorch Model Definitions ---
# NOTE: These should ideally match the definitions used in the training script
# (python/train_from_go_examples.py) to ensure compatibility.
# Re-defining them here for simplicity, but consider refactoring to a shared module.

class PyTorchPolicyNet(nn.Module):
    """MLP Policy Network."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        # Using LogSoftmax to match the training script's output for KLDivLoss
        # If ONNX consumer expects raw logits or probabilities, adjust this.
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.log_softmax(x)
        return x

class PyTorchValueNet(nn.Module):
    """MLP Value Network."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1) # Output is a single value
        self.tanh = nn.Tanh() # Output range [-1, 1] matching training script

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.tanh(x)
        return x

# --- Conversion Functions ---

def load_go_json_and_convert(json_path, onnx_path, model_type_str):
    """
    Loads model from Go JSON, loads weights into PyTorch model, exports to ONNX.
    (Handles the original functionality)
    """
    abs_json_path = os.path.abspath(json_path)
    print(f"Loading Go model from JSON: {abs_json_path}")
    with open(abs_json_path, 'r') as f:
        model_config = json.load(f)

    input_size = model_config["inputSize"]
    hidden_size = model_config["hiddenSize"]

    if model_type_str == "policy":
        output_size = model_config.get("outputSize", 9) # Default policy output size
        pytorch_model = PyTorchPolicyNet(input_size, hidden_size, output_size)
        print(f"Instantiated PyTorchPolicyNet (In: {input_size}, Hidden: {hidden_size}, Out: {output_size})")
        # --- Load Policy Weights (Assuming JSON structure) ---
        # Example keys - adjust based on your actual Go JSON export format
        if 'weightsInputHiddenPolicy' in model_config and 'biasesHiddenPolicy' in model_config and \
           'weightsHiddenOutputPolicy' in model_config and 'biasesOutputPolicy' in model_config:
           
            print("Loading Policy weights...")
            # Layer 1
            w1 = np.array(model_config['weightsInputHiddenPolicy']).astype(np.float32)
            b1 = np.array(model_config['biasesHiddenPolicy']).astype(np.float32)
            pytorch_model.layer1.weight.data = torch.from_numpy(w1)
            pytorch_model.layer1.bias.data = torch.from_numpy(b1)
            # Layer 2
            w2 = np.array(model_config['weightsHiddenOutputPolicy']).astype(np.float32)
            b2 = np.array(model_config['biasesOutputPolicy']).astype(np.float32)
            pytorch_model.layer2.weight.data = torch.from_numpy(w2)
            pytorch_model.layer2.bias.data = torch.from_numpy(b2)
            print("Policy weights loaded from JSON.")
        else:
            print("Warning: Policy weight keys not found in JSON. Model will have initial random weights.")

    elif model_type_str == "value":
        output_size = 1
        pytorch_model = PyTorchValueNet(input_size, hidden_size)
        print(f"Instantiated PyTorchValueNet (In: {input_size}, Hidden: {hidden_size}, Out: {output_size})")
        # --- Load Value Weights (Existing Logic) ---
        print("Loading Value weights...")
        # Layer 1 (fc1 in original script)
        w1 = np.array(model_config['weightsInputHidden']).astype(np.float32)
        b1 = np.array(model_config['biasesHidden']).astype(np.float32)
        pytorch_model.layer1.weight.data = torch.from_numpy(w1)
        pytorch_model.layer1.bias.data = torch.from_numpy(b1)
        # Layer 2 (fc2 in original script)
        w2 = np.array(model_config['weightsHiddenOutput']).astype(np.float32)
        # Go JSON 'biasOutput' is scalar, PyTorch bias is vector [1]
        b2 = torch.tensor([model_config['biasOutput']], dtype=torch.float32)
        pytorch_model.layer2.weight.data = torch.from_numpy(w2)
        pytorch_model.layer2.bias.data = b2
        print("Value weights loaded from JSON.")

    else:
        raise ValueError(f"Unknown model_type_str: {model_type_str}")

    # --- Export to ONNX ---
    export_pytorch_model_to_onnx(pytorch_model, input_size, onnx_path, model_type_str)

def load_pytorch_pt_and_convert(pt_path, onnx_path, pytorch_model_type, input_size, hidden_size, policy_output_size):
    """
    Loads a trained PyTorch model state dict from a .pt file, 
    instantiates the model, and exports it to ONNX.
    """
    abs_pt_path = os.path.abspath(pt_path)
    print(f"Loading PyTorch model state from: {abs_pt_path}")
    
    if pytorch_model_type == "policy":
        if policy_output_size is None:
            raise ValueError("--policy_output_size is required for --pytorch_model_type policy")
        pytorch_model = PyTorchPolicyNet(input_size, hidden_size, policy_output_size)
        print(f"Instantiated PyTorchPolicyNet (In: {input_size}, Hidden: {hidden_size}, Out: {policy_output_size})")
    elif pytorch_model_type == "value":
        pytorch_model = PyTorchValueNet(input_size, hidden_size)
        print(f"Instantiated PyTorchValueNet (In: {input_size}, Hidden: {hidden_size}, Out: 1)")
    else:
        raise ValueError(f"Unknown pytorch_model_type: {pytorch_model_type}")

    try:
        # Load the state dictionary
        state_dict = torch.load(abs_pt_path, map_location=torch.device('cpu')) # Load to CPU
        pytorch_model.load_state_dict(state_dict)
        print(f"Successfully loaded state dict from {abs_pt_path}")
    except Exception as e:
        print(f"Error loading state dict from {abs_pt_path}: {e}")
        raise

    # --- Export to ONNX ---
    export_pytorch_model_to_onnx(pytorch_model, input_size, onnx_path, pytorch_model_type)

def export_pytorch_model_to_onnx(pytorch_model, input_size, onnx_path, model_type_str):
    """Exports a given PyTorch model to ONNX format."""
    pytorch_model.eval() # Set to evaluation mode

    # Dummy input for ONNX export - Batch size 1
    dummy_input = torch.randn(1, input_size)
    
    abs_onnx_path = os.path.abspath(onnx_path)
    print(f"Exporting {model_type_str} model to ONNX: {abs_onnx_path}")
    
    try:
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            abs_onnx_path,
            input_names=['input'], # Standard name
            output_names=['output'], # Standard name
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11 # Use a reasonable default opset version
        )
        print(f"Model successfully exported to {abs_onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        raise

# --- Main Execution --- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Go JSON or PyTorch .pt models to ONNX.")
    
    # --- Mode Selection ---
    parser.add_argument(
        "--input_mode",
        type=str,
        required=True,
        choices=["go_json", "pytorch_pt"],
        help="Specify the input model format."
    )
    
    # --- Common Arguments ---
    parser.add_argument(
        "--onnx_output_path",
        type=str,
        required=True,
        help="Path for the output ONNX model file."
    )

    # --- Go JSON Specific Arguments ---
    parser.add_argument(
        "--go_json_input_path",
        type=str,
        help="Path to the input Go JSON model file (required if input_mode is go_json)."
    )
    parser.add_argument(
        "--go_model_type",
        type=str,
        choices=["value", "policy"],
        help="Type of the Go model to convert (required if input_mode is go_json)."
    )

    # --- PyTorch .pt Specific Arguments ---
    parser.add_argument(
        "--pytorch_input_path",
        type=str,
        help="Path to the input PyTorch .pt model file (required if input_mode is pytorch_pt)."
    )
    parser.add_argument(
        "--pytorch_model_type",
        type=str,
        choices=["value", "policy"],
        help="Type of the PyTorch model to convert (required if input_mode is pytorch_pt)."
    )
    parser.add_argument(
        "--input_size", 
        type=int, 
        help="Input layer size (required for pytorch_pt mode)."
    )
    parser.add_argument(
        "--hidden_size", 
        type=int, 
        help="Hidden layer size (required for pytorch_pt mode)."
    )
    parser.add_argument(
        "--policy_output_size", 
        type=int, 
        help="Output layer size for policy network (required for pytorch_pt mode with policy type)."
    )

    args = parser.parse_args()

    # --- Argument Validation based on Mode ---
    if args.input_mode == "go_json":
        if not args.go_json_input_path or not args.go_model_type:
            parser.error("--go_json_input_path and --go_model_type are required when --input_mode='go_json'")
        # Set common variables for printing
        input_path_for_print = args.go_json_input_path
        model_type_for_print = args.go_model_type
        
    elif args.input_mode == "pytorch_pt":
        if not args.pytorch_input_path or not args.pytorch_model_type or \
           args.input_size is None or args.hidden_size is None:
            parser.error("--pytorch_input_path, --pytorch_model_type, --input_size, and --hidden_size are required when --input_mode='pytorch_pt'")
        if args.pytorch_model_type == "policy" and args.policy_output_size is None:
             parser.error("--policy_output_size is required for --input_mode='pytorch_pt' and --pytorch_model_type='policy'")
        # Set common variables for printing
        input_path_for_print = args.pytorch_input_path
        model_type_for_print = args.pytorch_model_type
        
    else:
         # Should not happen due to choices constraint, but good practice
         parser.error(f"Invalid --input_mode: {args.input_mode}") 

    # Ensure output directory exists
    abs_onnx_output_path = os.path.abspath(args.onnx_output_path)
    output_dir = os.path.dirname(abs_onnx_output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    print(f"\n=== Starting ONNX Conversion ===")
    print(f"  Input Mode:   {args.input_mode}")
    print(f"  Input Path:   {os.path.abspath(input_path_for_print)}")
    print(f"  Model Type:   {model_type_for_print}")
    print(f"  Output ONNX:  {abs_onnx_output_path}")
    if args.input_mode == "pytorch_pt":
        print(f"  Input Size:   {args.input_size}")
        print(f"  Hidden Size:  {args.hidden_size}")
        if args.pytorch_model_type == "policy":
            print(f"  Policy Out:   {args.policy_output_size}")
    print("-------------------------------")

    # --- Call appropriate conversion function ---
    try:
        if args.input_mode == "go_json":
            load_go_json_and_convert(args.go_json_input_path, args.onnx_output_path, args.go_model_type)
        elif args.input_mode == "pytorch_pt":
            load_pytorch_pt_and_convert(
                args.pytorch_input_path, 
                args.onnx_output_path, 
                args.pytorch_model_type, 
                args.input_size, 
                args.hidden_size, 
                args.policy_output_size # Pass None if value type
            )
        print("\nConversion process completed successfully.")
        print(f"ONNX model saved to: {abs_onnx_output_path}")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        # Optionally re-raise if you want the script to exit with non-zero status
        # raise e 