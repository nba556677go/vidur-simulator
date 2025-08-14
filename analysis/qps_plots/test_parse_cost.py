import os
import csv
import tempfile

def test_get_device_costs():
    """
    Parse the Plannable_Public EC2_US East.csv to get hourly costs for different device types
    and include GPUs per node information
    """
    # Initialize with both cost and GPU counts
    device_costs = {
        'a100': {'cost': None, 'gpus_per_node': 8},  # p4d.24xlarge has 8 A100 GPUs
        'h100': {'cost': None, 'gpus_per_node': 8},  # p5.48xlarge has 8 H100 GPUs
        'l40s_g6e': {'cost': None, 'gpus_per_node': 8},  # g6e.48xlarge has 8 L40S GPUs
        'a10g_g5': {'cost': None, 'gpus_per_node': 8},  # g5.48xlarge has 8 A10G GPUs
        'l4_g6': {'cost': None, 'gpus_per_node': 8},  # g6.48xlarge has 8 L4 GPUs
    }
    
    # Instance type to device mapping
    instance_to_device = {
        'p4d.24xlarge': 'a100',
        'p5.48xlarge': 'h100',
        'g6e.48xlarge': 'l40s_g6e',
        'g5.48xlarge': 'a10g_g5',
        'g6.48xlarge': 'l4_g6'
    }
    
    csv_path = os.path.join(os.path.dirname(__file__), "Plannable_Public_EC2_US_East.csv")
    
    with open(csv_path, 'r') as f:
        # Skip the first line which contains 'sep=;'
        f.readline()
        
        # The file uses semicolons as separators
        reader = csv.DictReader(f, delimiter=';')
        
        for row in reader:
            instance_type = row.get('Instance Type', '').strip()
            print(f'Checking instance type: {instance_type}')
            
            if instance_type in instance_to_device:
                device = instance_to_device[instance_type]
                
                # Get the 2025 IMR cost (hourly rate)
                try:
                    imr_cost_str = row.get('2025 IMR', '0').strip()
                    if imr_cost_str:  # Make sure it's not empty
                        cost = float(imr_cost_str)
                        device_costs[device]['cost'] = cost
                        print(f'Found cost for {device} ({instance_type}): ${cost:.2f}')
                except (ValueError, TypeError) as e:
                    print(f'Error parsing cost for {instance_type}: {e}')
    
    # Print the costs and set fallback values if needed
    print("Device information:")
    fallback_costs = {
        'a100': 4.73,  # p4d.24xlarge approximate cost
        'h100': 11.85,  # p5.48xlarge approximate cost
        'l40s_g6e': 4.68,  # g6e.48xlarge approximate cost
        #'a10g_g5': 3.89,  # g5.48xlarge approximate cost
        #'l4_g6': 2.95  # g6.48xlarge approximate cost
    }
    
    for device, info in device_costs.items():
        # Use fallback values if costs weren't found
        if info['cost'] is None:
            info['cost'] = fallback_costs.get(device, 1.0)
            print(f"  {device}: ${info['cost']:.2f} per hour (FALLBACK VALUE), {info['gpus_per_node']} GPUs per node")
        else:
            print(f"  {device}: ${info['cost']:.2f} per hour, {info['gpus_per_node']} GPUs per node")
        
    return device_costs

if __name__ == "__main__":
    test_get_device_costs()
