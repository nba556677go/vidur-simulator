import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

class ConfigOptimizerPlotter:
    """
    A class to plot configuration optimizer results, focusing on max QPS per dollar
    for each parallelism/replica configuration under different network devices.
    """
    
    def __init__(self, csv_file, input_param=None, output_dir="max_qps"):
        """
        Initialize the plotter with CSV data and input parameters.
        
        Args:
            csv_file (str): Path to the CSV file containing config optimizer results
            input_param (dict): Dictionary with filtering parameters like 
                               {'prefill_tokens': 300, 'decode_tokens': 3}
            output_dir (str): Directory to save the plots (default: "max_qps")
        """
        self.csv_file = csv_file
        self.input_param = input_param or {'prefill_tokens': 300, 'decode_tokens': 3}
        self.output_dir = output_dir
        self.df = None
        self.filtered_df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # SLO limits
        self.slo_limit = 0.25  # 250ms for TTFT
        self.exec_slo = 7.8    # slo for total execution time
        self.inter_token_slo = 0.015  # 15ms in seconds
        
    def load_and_filter_data(self):
        """Load CSV data and apply input parameter filters."""
        print(f"Loading data from {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        
        print(f"Original data shape: {self.df.shape}")
        
        # Apply input parameter filters
        filter_conditions = []
        for param, value in self.input_param.items():
            if param in self.df.columns:
                filter_conditions.append(self.df[param] == value)
                print(f"Filtering by {param} = {value}")
            else:
                print(f"Warning: Column {param} not found in data")
        
        if filter_conditions:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            self.filtered_df = self.df[combined_filter].copy()
        else:
            self.filtered_df = self.df.copy()
            
        print(f"Filtered data shape: {self.filtered_df.shape}")
        
        if len(self.filtered_df) == 0:
            print("Warning: No data remains after filtering!")
            return
            
        # Print unique models found
        unique_models = self.filtered_df['model_name'].unique()
        print(f"Unique models found: {unique_models}")
        
    def get_max_qps_per_dollar_data(self, model_df):
        """
        For each unique combination of parallelism parameters and network_device,
        find the configuration with maximum QPS per dollar.
        
        Args:
            model_df (DataFrame): Data for a specific model
            
        Returns:
            DataFrame: Data with max QPS per dollar for each configuration group
        """
        # Group by parallelism parameters and network device
        groupby_cols = ['network_device', 'tensor_parallel_size', 'num_pipeline_stages', 'num_replicas']
        
        # Find the row with max qps_per_dollar for each group
        max_qps_per_dollar_data = []
        
        for group_keys, group_df in model_df.groupby(groupby_cols):
            if len(group_df) > 0:
                # Find the row with maximum qps_per_dollar in this group
                max_idx = group_df['qps_per_dollar'].idxmax()
                max_row = group_df.loc[max_idx].copy()
                max_qps_per_dollar_data.append(max_row)
        
        return pd.DataFrame(max_qps_per_dollar_data)
    
    def create_subplots_for_model(self, model_name, model_data, plot_type='qps'):
        """
        Create the 4-subplot figure for a specific model.
        
        Args:
            model_name (str): Name of the model
            model_data (DataFrame): Filtered data for this model
            plot_type (str): Either 'qps' or 'qps_per_dollar'
        """
        # Get max QPS per dollar data
        max_data = self.get_max_qps_per_dollar_data(model_data)
        
        if len(max_data) == 0:
            print(f"No data found for model {model_name}")
            return
            
        print(f"Plotting {len(max_data)} max QPS per dollar points for model {model_name}")
        
        # Create color mapping for network devices
        unique_network_devices = max_data['network_device'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_network_devices), 1)))
        color_map = {device: colors[i] for i, device in enumerate(unique_network_devices)}
        
        # Determine y-axis values and titles based on plot type
        if plot_type == 'qps_per_dollar':
            y_col = 'qps_per_dollar'
            y_label = 'QPS per Dollar'
            fig_title_prefix = 'Cost Efficiency'
            color_metric = 'qps_per_dollar'
            color_label = 'QPS per Dollar'
        else:
            y_col = 'qps'
            y_label = 'QPS'
            fig_title_prefix = 'Performance'
            color_metric = 'qps'
            color_label = 'QPS'
        
        # Find best configs
        slo_compliant = max_data[(max_data['p99_ttft'] <= self.slo_limit) & 
                                (max_data['p99_exec_time'] <= self.exec_slo)]
        
        if len(slo_compliant) > 0:
            if plot_type == 'qps_per_dollar':
                best_config = slo_compliant.loc[slo_compliant['qps_per_dollar'].idxmax()]
            else:   
                best_config = slo_compliant.loc[slo_compliant['qps'].idxmax()]
        else:
            print(f"Warning: No SLO compliant configs for {model_name}, using best TTFT")
            best_config = max_data.loc[max_data['p99_ttft'].idxmin()]
        
        # Create best config description
        best_desc = (f"Best {y_label} Config for {model_name}: "
                    f"PP={best_config['num_pipeline_stages']}, "
                    f"TP={best_config['tensor_parallel_size']}, "
                    f"Replicas={best_config['num_replicas']}, "
                    f"Nodes={best_config['nodes_needed']}, "
                    f"Scheduler={best_config['scheduler_type']}, "
                    f"SKU={best_config['network_device']}, "
                    f"QPS={best_config['qps']:.2f}, "
                    f"QPS/$={best_config['qps_per_dollar']:.4f}")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), 
                                gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
        fig.text(0.5, 0.97, best_desc, ha='center', fontsize=17)
        
        # Check data availability
        has_exec_time = all(pd.notna(max_data['p99_exec_time']))
        has_inter_token_latency = pd.notna(max_data['p99_inter_token_latency']).any()
        
        # Subplot 1: Y vs P99 TTFT
        ax = axes[0, 0]
        ax.set_title(f"{y_label} vs P99 Time to First Token", fontsize=19)
        
        for device in unique_network_devices:
            subset = max_data[max_data['network_device'] == device]
            ax.scatter(subset['p99_ttft'], subset[y_col], 
                      label=device, color=color_map[device], s=80, alpha=0.7)
        
        ax.axvline(x=self.slo_limit, color='red', linestyle='--', 
                  label=f'SLO Limit ({self.slo_limit*1000}ms)')
        ax.axvspan(0, self.slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel(y_label, fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Network Device", fontsize=13)
        
        # Subplot 2: Y vs P99 Request Total Latency
        ax = axes[0, 1]
        ax.set_title(f"{y_label} vs P99 Request Total Latency", fontsize=19)
        
        if has_exec_time:
            for device in unique_network_devices:
                subset = max_data[max_data['network_device'] == device]
                ax.scatter(subset['p99_exec_time'], subset[y_col], 
                          label=device, color=color_map[device], s=80, alpha=0.7)
            
            ax.axvline(x=self.exec_slo, color='red', linestyle='--', 
                      label=f'SLO Limit ({self.exec_slo}s)')
            ax.axvspan(0, self.exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
            
            ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
            ax.set_ylabel(y_label, fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Network Device", fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Subplot 3: Y vs P99 Inter-Token Latency
        ax = axes[1, 0]
        ax.set_title(f"{y_label} vs P99 Inter-Token Latency", fontsize=19)
        
        if has_inter_token_latency:
            for device in unique_network_devices:
                subset = max_data[max_data['network_device'] == device]
                ax.scatter(subset['p99_inter_token_latency'], subset[y_col], 
                          label=device, color=color_map[device], s=80, alpha=0.7)
            
            ax.axvline(x=self.inter_token_slo, color='red', linestyle='--', 
                      label=f'Inter-Token SLO ({self.inter_token_slo*1000}ms)')
            ax.axvspan(0, self.inter_token_slo, alpha=0.1, color='green', 
                      label='SLO Compliant Region')
            
            ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
            ax.set_ylabel(y_label, fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Network Device", fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Subplot 4: P99 Inter-Token Latency vs P99 TTFT, colored by metric
        ax = axes[1, 1]
        ax.set_title(f"P99 Inter-Token Latency vs P99 TTFT (Colored by {color_label})", fontsize=19)
        
        if has_inter_token_latency:
            scatter = ax.scatter(max_data['p99_ttft'], max_data['p99_inter_token_latency'], 
                               c=max_data[color_metric], cmap='viridis', s=100, alpha=0.7)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label(color_label, fontsize=17)
            
            # Highlight best config
            ax.scatter(best_config['p99_ttft'], best_config['p99_inter_token_latency'], 
                      color='gold', s=250, marker='*', 
                      label=f"Best: {best_config['network_device']} TP{best_config['tensor_parallel_size']}/PP{best_config['num_pipeline_stages']}", 
                      edgecolor='black', zorder=10)
            
            # Add SLO limit lines
            ax.axvline(x=self.slo_limit, color='red', linestyle='--', 
                      label=f'TTFT SLO ({self.slo_limit*1000}ms)')
            ax.axhline(y=self.inter_token_slo, color='red', linestyle=':', 
                      label=f'Inter-Token SLO ({self.inter_token_slo*1000}ms)')
            
            # Add SLO compliant region
            ax.fill_between([0, self.slo_limit], 0, self.inter_token_slo, 
                           alpha=0.1, color='green', label='SLO Compliant Region')
            
            ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
            ax.set_ylabel("Inter-Token Latency - P99 (s)", fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Clean model name for filename
        clean_model_name = model_name.replace('/', '_').replace(' ', '_')
        filename = f"max_qps_per_dollar_{plot_type}_{clean_model_name}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
        
    def plot_all_models(self):
        """Create plots for all models in the filtered data."""
        if self.filtered_df is None:
            print("No data loaded. Call load_and_filter_data() first.")
            return
            
        unique_models = self.filtered_df['model_name'].unique()
        
        for model_name in unique_models:
            print(f"\nProcessing model: {model_name}")
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            
            # Create both QPS and QPS per dollar plots
            self.create_subplots_for_model(model_name, model_data, plot_type='qps')
            self.create_subplots_for_model(model_name, model_data, plot_type='qps_per_dollar')
    
    def print_summary(self):
        """Print summary statistics of the filtered data."""
        if self.filtered_df is None:
            print("No data loaded.")
            return
            
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total configurations: {len(self.filtered_df)}")
        print(f"Input parameters: {self.input_param}")
        
        print(f"\nUnique models: {list(self.filtered_df['model_name'].unique())}")
        print(f"Unique network devices: {list(self.filtered_df['network_device'].unique())}")
        
        # Summary by model
        for model_name in self.filtered_df['model_name'].unique():
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            max_data = self.get_max_qps_per_dollar_data(model_data)
            
            print(f"\n{model_name}:")
            print(f"  Total configs: {len(model_data)}")
            print(f"  Max QPS per dollar points: {len(max_data)}")
            if len(max_data) > 0:
                print(f"  Best QPS per dollar: {max_data['qps_per_dollar'].max():.4f}")
                print(f"  Best QPS: {max_data['qps'].max():.2f}")


def main():
    """Main function to run the plotter."""
    # Configuration
    csv_file = "config_optimizer_results.csv"
    input_param = {'prefill_tokens': 300, 'decode_tokens': 3}
    
    # Create plotter instance
    plotter = ConfigOptimizerPlotter(csv_file, input_param)
    
    # Load and filter data
    plotter.load_and_filter_data()
    
    # Print summary
    plotter.print_summary()
    
    # Create plots for all models
    plotter.plot_all_models()
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
