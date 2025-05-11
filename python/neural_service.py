#!/usr/bin/env python3
import os
import sys
import time
import argparse
import logging
import platform
import numpy as np
import tensorflow as tf
import grpc
from concurrent import futures

# Add the proto directory to the path so we can import the generated modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proto.neural_service_pb2 as neural_pb2
import proto.neural_service_pb2_grpc as neural_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if running on Apple Silicon
is_apple_silicon = (platform.system() == 'Darwin' and platform.machine() == 'arm64')
if is_apple_silicon:
    logger.info("Detected Apple Silicon. Optimizing for Metal GPU.")
    # Explicitly enable Metal API
    os.environ['DEVICE_NAME'] = 'metal'

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"Found {len(physical_devices)} GPUs: {physical_devices}")
    # Configure TensorFlow to use memory growth to avoid allocating all GPU memory at once
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Memory growth enabled for GPUs")
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.warning("No GPU found, using CPU")

# If on Apple Silicon, verify Metal is being used
if is_apple_silicon:
    # This will force TensorFlow to initialize and print out device placements
    with tf.device('/device:GPU:0'):
        try:
            tf.random.normal([1])
            logger.info("Successfully initialized Metal GPU.")
        except Exception as e:
            logger.error(f"Failed to use Metal GPU: {e}")

class RPSNeuralNetwork:
    """Neural network implementation for RPS game"""
    
    def __init__(self, input_size, hidden_size, output_size, model_type="policy"):
        """Initialize neural network with the specified architecture"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_type = model_type
        
        # Build and compile the model
        self.model = self.build_model()
        self.device = "gpu" if physical_devices else "cpu"
        logger.info(f"Created {model_type} model with {input_size} inputs, {hidden_size} hidden, "
                   f"{output_size} outputs on {self.device}")
    
    def build_model(self):
        """Build the TensorFlow model architecture"""
        # Use Metal GPU if available on Apple Silicon
        device_strategy = '/device:GPU:0' if physical_devices else '/CPU:0'
        
        with tf.device(device_strategy):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.input_size,)),
                tf.keras.layers.Dense(self.hidden_size, activation='relu'),
                tf.keras.layers.Dense(self.output_size, 
                                      activation='softmax' if self.model_type == "policy" else 'tanh')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy' if self.model_type == "policy" else 'mse'
            )
        
        return model
    
    def predict(self, features):
        """Run inference on a single input"""
        input_array = np.array(features).reshape(1, -1)
        result = self.model.predict(input_array, verbose=0)
        
        if self.model_type == "policy":
            best_move = np.argmax(result[0])
            return result[0], 0.0, best_move
        else:  # value network
            return [], float(result[0][0]), 0
    
    def batch_predict(self, batch_features):
        """Run inference on a batch of inputs"""
        # Convert list of feature lists to numpy array
        input_array = np.array(batch_features)
        results = self.model.predict(input_array, verbose=0)
        
        outputs = []
        
        if self.model_type == "policy":
            for i in range(len(batch_features)):
                best_move = np.argmax(results[i])
                outputs.append((results[i], 0.0, best_move))
        else:  # value network
            for i in range(len(batch_features)):
                outputs.append(([], float(results[i][0]), 0))
                
        return outputs
    
    def load_weights(self, weights_path):
        """Load weights from a saved file"""
        try:
            self.model.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return False

class NeuralServicer(neural_pb2_grpc.NeuralServiceServicer):
    """gRPC servicer implementation for neural network inference"""
    
    def __init__(self):
        """Initialize service with policy and value networks"""
        # Default network configurations
        self.policy_net = RPSNeuralNetwork(
            input_size=64,  # Default size, will be updated when loading model
            hidden_size=128,
            output_size=8,
            model_type="policy"
        )
        
        self.value_net = RPSNeuralNetwork(
            input_size=64,  # Default size, will be updated when loading model 
            hidden_size=128,
            output_size=1,
            model_type="value"
        )
        
        # Performance metrics
        self.total_requests = 0
        self.total_batch_size = 0
        self.inference_time = 0
        self.start_time = time.time()
    
    def Predict(self, request, context):
        """Handle single prediction request"""
        start_time = time.time()
        self.total_requests += 1
        
        # Validate input
        if len(request.features) == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty features provided")
            return neural_pb2.PredictResponse()
        
        # Select appropriate network
        network = self.policy_net if request.model_type == "policy" else self.value_net
        
        # Run prediction
        probabilities, value, best_move = network.predict(request.features)
        
        # Measure performance
        self.inference_time += time.time() - start_time
        
        # Create response
        response = neural_pb2.PredictResponse()
        
        if network.model_type == "policy":
            response.probabilities.extend(probabilities)
        else:
            response.value = value
            
        response.best_move = best_move
        
        return response
    
    def BatchPredict(self, request, context):
        """Handle batch prediction request"""
        start_time = time.time()
        batch_size = len(request.inputs)
        self.total_requests += 1
        self.total_batch_size += batch_size
        
        # Validate input
        if batch_size == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty batch provided")
            return neural_pb2.BatchPredictResponse()
        
        # Select appropriate network
        network = self.policy_net if request.model_type == "policy" else self.value_net
        
        # Prepare batch features
        batch_features = [list(input_features.features) for input_features in request.inputs]
        
        # Run batch prediction
        batch_results = network.batch_predict(batch_features)
        
        # Measure performance
        self.inference_time += time.time() - start_time
        
        # Create response
        response = neural_pb2.BatchPredictResponse()
        
        for probabilities, value, best_move in batch_results:
            output = neural_pb2.PredictResponse()
            
            if network.model_type == "policy":
                output.probabilities.extend(probabilities)
            else:
                output.value = value
                
            output.best_move = best_move
            response.outputs.append(output)
        
        return response
    
    def GetModelInfo(self, request, context):
        """Provide information about the loaded model"""
        # Select appropriate network
        network = self.policy_net if request.model_type == "policy" else self.value_net
        
        # Create response
        response = neural_pb2.ModelInfoResponse()
        response.input_size = network.input_size
        response.hidden_size = network.hidden_size
        response.output_size = network.output_size
        response.device = "metal" if (is_apple_silicon and physical_devices) else network.device
        response.framework = "tensorflow"
        
        return response
    
    def print_stats(self):
        """Print performance statistics"""
        elapsed = time.time() - self.start_time
        
        if self.total_requests > 0:
            avg_inference = (self.inference_time * 1000) / self.total_requests
            avg_batch_size = self.total_batch_size / self.total_requests if self.total_batch_size > 0 else 0
            
            logger.info(f"Stats: {self.total_requests} requests, avg inference: {avg_inference:.2f}ms, "
                       f"avg batch: {avg_batch_size:.1f}, uptime: {elapsed:.1f}s")

def serve(port, policy_weights=None, value_weights=None, max_workers=10):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = NeuralServicer()
    
    # Load weights if provided
    if policy_weights and os.path.exists(policy_weights):
        servicer.policy_net.load_weights(policy_weights)
    
    if value_weights and os.path.exists(value_weights):
        servicer.value_net.load_weights(value_weights)
    
    neural_pb2_grpc.add_NeuralServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    device_info = "Metal GPU" if (is_apple_silicon and physical_devices) else servicer.policy_net.device
    logger.info(f"Neural service started on port {port}")
    logger.info(f"Using device: {device_info}")
    
    try:
        # Print stats periodically
        while True:
            time.sleep(60)
            servicer.print_stats()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network gRPC service for RPS")
    parser.add_argument("--port", type=int, default=50052,
                       help="Port for the gRPC server")
    parser.add_argument("--policy-weights", type=str, default=None,
                       help="Path to policy network weights")
    parser.add_argument("--value-weights", type=str, default=None,
                       help="Path to value network weights")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of worker threads")
    
    args = parser.parse_args()
    
    serve(args.port, args.policy_weights, args.value_weights, args.workers) 