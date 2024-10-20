def ffd_bin_packing(weights, capacity):
    # Combine weights with their indices
    weights = list(enumerate(weights))
    
    # Sort weights in decreasing order
    weights.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize bins and bin_sums
    bins = []
    bin_sums = []
    
    for index, weight in weights:
        # Try to fit weight into existing bins
        for i, bin_sum in enumerate(bin_sums):
            if bin_sum + weight <= capacity:
                bins[i].append((index, weight))
                bin_sums[i] += weight
                break
        else:
            # If weight can't fit into any existing bin, create a new bin
            bins.append([(index, weight)])
            bin_sums.append(weight)
            
    return bins

weights = [10, 20, 30, 40, 50, 60, 70, 80]
capacity = 100

bins = ffd_bin_packing(weights, capacity)

for i, bin in enumerate(bins):
    print(f"Bin {i+1}: {bin}, Total weight: {sum(weight for index, weight in bin)}")
