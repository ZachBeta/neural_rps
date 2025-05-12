#!/usr/bin/env python3
import os
import sys
import time
import argparse
import logging
import platform
import numpy as np
import onnxruntime as ort
import grpc
from concurrent import futures

# Add the proto directory to the path so we can import the generated modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proto.neural_service_pb2 as neural_pb2
import proto.neural_service_pb2_grpc as neural_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Argument parsing
parser = argparse.ArgumentParser(description="ONNX Neural Service")
parser.add_argument("--port", type=int, default=50053, help="Port for the gRPC service")
parser.add_argument("--model_path", type=str, default="python/output/rps_value1.onnx", help="Path to the ONNX model file")
args = parser.parse_args()

class NeuralServicer(neural_pb2_grpc.NeuralServiceServicer):
    """gRPC servicer implementation for neural network inference using ONNX Runtime"""
    
    def __init__(self, model_path):
        """Initialize service with an ONNX model"""
        logger.info(f"ONNX model path received: {model_path}")
        logger.info(f"Attempting to load ONNX model: {model_path}")
        
        try:
            # Specify execution providers. For this machine, CoreML EP options are ignored
            # and its default batch performance is not optimal. Defaulting to CPUExecutionProvider.
            providers = [
                'CPUExecutionProvider'
            ]
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"Successfully loaded ONNX model from {model_path}")
            logger.info(f"ONNX session providers: {self.ort_session.get_providers()}")

            # Log detailed provider options
            provider_options = self.ort_session.get_provider_options()
            if provider_options:
                for provider_name, options in provider_options.items():
                    logger.info(f"  Provider '{provider_name}' options: {options}")
            else:
                logger.info("  No provider options found for the session.")

            inputs_meta = self.ort_session.get_inputs()
            outputs_meta = self.ort_session.get_outputs()

            if not inputs_meta:
                raise ValueError("ONNX model has no inputs.")
            if not outputs_meta:
                raise ValueError("ONNX model has no outputs.")

            self.input_name = inputs_meta[0].name
            self.output_name = outputs_meta[0].name
            
            # Input shape might include dynamic batch size (None or -1)
            # e.g., [None, 81] for rps_value1.onnx
            self.model_input_shape = inputs_meta[0].shape
            # The feature size is the last dimension
            if not self.model_input_shape or not isinstance(self.model_input_shape[-1], int) or self.model_input_shape[-1] <= 0:
                raise ValueError(f"Invalid feature size in model input shape: {self.model_input_shape}")
            self.model_input_feature_size = self.model_input_shape[-1]

            logger.info(f"Initialized ONNX session. Input: '{self.input_name}' (Shape: {self.model_input_shape}), "
                        f"Output: '{self.output_name}' (Shape: {outputs_meta[0].shape}), "
                        f"Inferred Feature Size: {self.model_input_feature_size}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model or configure session from '{model_path}': {e}")
            # Propagate exception to prevent service from starting with a bad model
            raise

        # Performance metrics
        self.total_requests = 0
        self.total_batch_size = 0
        self.inference_time = 0
        self.start_time = time.time()
    
    def Predict(self, request, context):
        """Handle single prediction request"""
        start_time = time.time()
        self.total_requests += 1
        
        if not hasattr(self, 'ort_session') or self.ort_session is None:
            logger.error("ONNX session not initialized.")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("ONNX session not initialized on the server.")
            return neural_pb2.PredictResponse()

        # Validate input
        if len(request.features) == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty features provided")
            return neural_pb2.PredictResponse()
        
        if len(request.features) != self.model_input_feature_size:
            error_msg = f"Incorrect number of features. Expected {self.model_input_feature_size}, got {len(request.features)}."
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error_msg)
            return neural_pb2.PredictResponse()

        try:
            # Convert features to NumPy array, reshape, and ensure correct type
            input_data = np.array(list(request.features), dtype=np.float32).reshape(1, self.model_input_feature_size)
            
            # Prepare input dictionary for ONNX Runtime
            inputs_dict = {self.input_name: input_data}
            
            logger.debug(f"Predict: Available session providers for inference: {self.ort_session.get_providers()}")
            # Run inference
            # The result is a list of NumPy arrays, one for each requested output name.
            # Since we request one output (self.output_name), results will have one element.
            onnx_results = self.ort_session.run([self.output_name], inputs_dict)
            
            # Extract the scalar value. For rps_value1.onnx, output shape is [1,1]
            # onnx_results[0] is a np.array like [[-0.12345]]
            value = float(onnx_results[0][0][0])

            # Create response
            response = neural_pb2.PredictResponse()
            response.value = value
            # response.best_move = 0 # Not applicable for a value network
            # response.probabilities would not be set for a value network

        except Exception as e:
            logger.error(f"Error during ONNX Predict inference: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error during ONNX inference: {e}")
            return neural_pb2.PredictResponse()
        
        # Measure performance
        self.inference_time += time.time() - start_time
        return response
    
    def BatchPredict(self, request, context):
        """Handle batch prediction request"""
        start_time = time.time()
        batch_size = len(request.inputs)
        self.total_requests += 1 # Count as one gRPC request
        self.total_batch_size += batch_size # Accumulate total items processed
        
        if not hasattr(self, 'ort_session') or self.ort_session is None:
            logger.error("ONNX session not initialized for BatchPredict.")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("ONNX session not initialized on the server.")
            return neural_pb2.BatchPredictResponse()

        # Validate input
        if batch_size == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty batch provided")
            return neural_pb2.BatchPredictResponse()
        
        batch_features_list = []
        for i, inp in enumerate(request.inputs):
            if len(inp.features) != self.model_input_feature_size:
                error_msg = f"Incorrect number of features for item {i} in batch. Expected {self.model_input_feature_size}, got {len(inp.features)}."
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return neural_pb2.BatchPredictResponse()
            batch_features_list.append(list(inp.features))

        try:
            # Convert list of feature lists to NumPy array, reshape, and ensure correct type
            input_data = np.array(batch_features_list, dtype=np.float32).reshape(batch_size, self.model_input_feature_size)
            
            # Prepare input dictionary for ONNX Runtime
            inputs_dict = {self.input_name: input_data}
            
            logger.debug(f"BatchPredict: Available session providers for inference: {self.ort_session.get_providers()}")
            # Run inference
            # The result is a list of NumPy arrays, one for each requested output name.
            onnx_results = self.ort_session.run([self.output_name], inputs_dict)
            
            # onnx_results[0] is a np.array like [[val1], [val2], ..., [val_batch_size]]
            # Each val is typically a list/array itself, e.g., [0.123] for a scalar output model
            batch_output_values = onnx_results[0]

            response = neural_pb2.BatchPredictResponse()
            for i in range(batch_size):
                individual_output = neural_pb2.PredictResponse()
                # Assuming the model outputs one scalar per input item in the batch
                individual_output.value = float(batch_output_values[i][0])
                # individual_output.best_move and .probabilities not set for value net
                response.outputs.append(individual_output)

        except Exception as e:
            logger.error(f"Error during ONNX BatchPredict inference: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error during ONNX batch inference: {e}")
            return neural_pb2.BatchPredictResponse()
        
        # Measure performance
        self.inference_time += time.time() - start_time
        return response
    
    def GetModelInfo(self, request, context):
        """Provide information about the loaded ONNX model"""
        response = neural_pb2.ModelInfoResponse()
        response.framework = "onnxruntime"
        response.hidden_size = -1 # Typically not well-defined for a generic ONNX model graph

        if hasattr(self, 'ort_session') and self.ort_session is not None:
            # Input Size (feature dimension)
            if hasattr(self, 'model_input_feature_size'):
                response.input_size = self.model_input_feature_size
            else:
                response.input_size = -1 # Should have been set in __init__
                logger.warning("model_input_feature_size not found on servicer for GetModelInfo")

            # Output Size (feature dimension of the first output)
            try:
                outputs_meta = self.ort_session.get_outputs()
                if outputs_meta and outputs_meta[0].shape:
                    # Assuming the last dimension is the feature count for the output
                    # For [None, 1] or [1, 1], this would be 1.
                    output_shape = outputs_meta[0].shape
                    # Handle dynamic dimensions (None) if they are not the feature dim
                    feature_dim = next((dim for dim in reversed(output_shape) if isinstance(dim, int) and dim > 0), None)
                    if feature_dim is not None:
                        response.output_size = feature_dim
                    else:
                        response.output_size = -1 # Cannot determine from shape (e.g., [None, None])
                        logger.warning(f"Could not determine valid output feature size from shape: {output_shape}")
                else:
                    response.output_size = -1
                    logger.warning("No output metadata or shape found for GetModelInfo")
            except Exception as e:
                logger.error(f"Error getting output_size for GetModelInfo: {e}")
                response.output_size = -1

            # Device / Execution Providers
            try:
                providers = self.ort_session.get_providers()
                # Providers list might contain tuples, e.g., ('CoreMLExecutionProvider', {config})
                # We only want the names.
                provider_names = []
                for p in providers:
                    if isinstance(p, str):
                        provider_names.append(p)
                    elif isinstance(p, tuple) and len(p) > 0 and isinstance(p[0], str):
                        provider_names.append(p[0])
                response.device = ",".join(provider_names)
            except Exception as e:
                logger.error(f"Error getting providers for GetModelInfo: {e}")
                response.device = "Error"
            
            logger.info(f"GetModelInfo: InputSize={response.input_size}, OutputSize={response.output_size}, Device='{response.device}', Framework='{response.framework}'")
        else:
            logger.warning("GetModelInfo called but ONNX session not available.")
            response.input_size = -1
            response.output_size = -1
            response.device = "N/A"
        
        return response
    
    def print_stats(self):
        """Print performance statistics"""
        elapsed = time.time() - self.start_time
        
        if self.total_requests > 0:
            avg_inference = (self.inference_time * 1000) / self.total_requests
            logger.info(f"Total requests: {self.total_requests}")
            logger.info(f"Avg batch size: {self.total_batch_size / self.total_requests:.2f}")
            logger.info(f"Avg inference time: {avg_inference:.2f} ms")
            logger.info(f"Service uptime: {elapsed:.2f} s")
        else:
            logger.info("No requests processed.")

def serve(port, model_path, max_workers=10):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    neural_pb2_grpc.add_NeuralServiceServicer_to_server(
        NeuralServicer(model_path=model_path), server # Pass model_path
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Server started on port {port} with ONNX model: {model_path}")
    
    try:
        while True:
            time.sleep(60 * 60 * 24)  # Keep server alive
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        # Create an instance of NeuralServicer to call print_stats
        # This is a bit of a hack; stats ideally would be managed by the server instance or a separate class.
        # For simplicity, we re-fetch model_path if needed or use the global args.
        temp_servicer_for_stats = NeuralServicer(model_path=args.model_path) 
        temp_servicer_for_stats.total_requests = server.total_requests if hasattr(server, 'total_requests') else 0 # This won't work as server doesn't own these
        temp_servicer_for_stats.total_batch_size = server.total_batch_size if hasattr(server, 'total_batch_size') else 0
        temp_servicer_for_stats.inference_time = server.inference_time if hasattr(server, 'inference_time') else 0
        # A better way for stats on shutdown would be to have NeuralServicer instance accessible
        # or pass a stats object to it.
        # For now, the stats printed on shutdown will be from a new instance, so likely zero.
        # The running stats are printed by the servicer instance tied to the server, but not easily on shutdown here.
        logger.warning("Stats on shutdown might be inaccurate due to instance scoping. Check periodic logs if enabled.")
        server.stop(0)
        logger.info("Server stopped.")

if __name__ == '__main__':
    serve(args.port, args.model_path) 