#!/usr/bin/env python3
import os
import sys
import time
import argparse
import logging
import numpy as np
import grpc

# Add the proto directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proto.neural_service_pb2 as neural_pb2
import proto.neural_service_pb2_grpc as neural_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_input(size):
    """Generate random input features"""
    return np.random.uniform(-1, 1, size).tolist()

def generate_random_batch(batch_size, input_size):
    """Generate a batch of random inputs"""
    return [generate_random_input(input_size) for _ in range(batch_size)]

def test_single_predict(stub, input_size, iterations, model_type="policy"):
    """Test single prediction performance"""
    input_features = generate_random_input(input_size)
    
    total_time = 0
    for i in range(iterations):
        start_time = time.time()
        
        request = neural_pb2.PredictRequest(
            features=input_features,
            model_type=model_type
        )
        
        response = stub.Predict(request)
        
        elapsed = time.time() - start_time
        total_time += elapsed
    
    avg_time = (total_time * 1000) / iterations  # ms
    logger.info(f"Single prediction: {iterations} iterations, avg time: {avg_time:.2f} ms")
    
    return avg_time

def test_batch_predict(stub, input_size, batch_size, iterations, model_type="policy"):
    """Test batch prediction performance"""
    batch_features = generate_random_batch(batch_size, input_size)
    
    total_time = 0
    for i in range(iterations):
        start_time = time.time()
        
        # Create batch request
        request = neural_pb2.BatchPredictRequest(
            model_type=model_type,
            inputs=[neural_pb2.InputFeatures(features=features) for features in batch_features]
        )
        
        response = stub.BatchPredict(request)
        
        elapsed = time.time() - start_time
        total_time += elapsed
    
    avg_time = (total_time * 1000) / iterations  # ms
    per_prediction = avg_time / batch_size
    logger.info(f"Batch prediction: {iterations} batches of {batch_size}, "
                f"avg batch time: {avg_time:.2f} ms, per prediction: {per_prediction:.2f} ms")
    
    return avg_time, per_prediction

def main():
    parser = argparse.ArgumentParser(description="Test neural service performance")
    parser.add_argument("--addr", default="localhost:50052", help="Service address")
    parser.add_argument("--input-size", type=int, default=64, help="Input size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--model-type", default="policy", choices=["policy", "value"], 
                       help="Model type (policy or value)")
    
    args = parser.parse_args()
    
    # Create gRPC channel
    channel = grpc.insecure_channel(args.addr)
    stub = neural_pb2_grpc.NeuralServiceStub(channel)
    
    try:
        # Get model info
        info_request = neural_pb2.ModelInfoRequest(model_type=args.model_type)
        info = stub.GetModelInfo(info_request)
        
        logger.info(f"Connected to neural service, model info:")
        logger.info(f"Input size: {info.input_size}, Hidden size: {info.hidden_size}, "
                   f"Output size: {info.output_size}")
        logger.info(f"Device: {info.device}, Framework: {info.framework}")
        
        print("\n===== Performance Test =====")
        
        # Test single prediction
        single_avg = test_single_predict(
            stub, args.input_size, args.iterations, args.model_type
        )
        
        # Test batch prediction
        batch_avg, per_prediction = test_batch_predict(
            stub, args.input_size, args.batch_size, args.iterations, args.model_type
        )
        
        # Calculate speedup
        speedup = (single_avg * args.batch_size) / batch_avg
        
        print(f"\nBatch speedup: {speedup:.2f}x faster than sequential prediction")
        print(f"With batch size {args.batch_size}, each prediction takes {per_prediction:.2f} ms")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()

if __name__ == "__main__":
    main() 