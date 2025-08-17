#!/usr/bin/env python3
import sys
import re
import numpy as np

def parse_benchmark_file(filename):
    throughput_values = []
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
            # Extract all request throughput values
            pattern = r'Request throughput \(req/s\):\s+(\d+\.\d+)'
            throughput_values = [float(match) for match in re.findall(pattern, content)]
            
        if not throughput_values:
            print(f"No request throughput values found in {filename}")
            sys.exit(1)
            
        return throughput_values
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse-fib.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    throughput_values = parse_benchmark_file(filename)
    
    # Drop the first throughput value
    first_value = throughput_values[0]
    throughput_values = throughput_values[1:]
    
    if not throughput_values:
        print("Not enough throughput values after dropping the first one")
        sys.exit(1)
    
    # Calculate statistics
    throughput_array = np.array(throughput_values)
    mean = np.mean(throughput_array)
    median = np.median(throughput_array)
    std_dev = np.std(throughput_array, ddof=1)  # ddof=1 for sample standard deviation
    std_err = std_dev / np.sqrt(len(throughput_array))
    std_err_percent = (std_err / mean) * 100
    
    # Print results
    print(f"Dropped first throughput value: {first_value:.2f}")
    print(f"Remaining throughput values: {', '.join(f'{val:.2f}' for val in throughput_values)}")
    print(f"Median throughput: {median:.2f} req/s")
    print(f"Standard error: {std_err:.2f} ({std_err_percent:.2f}%)")

if __name__ == "__main__":
    main()
