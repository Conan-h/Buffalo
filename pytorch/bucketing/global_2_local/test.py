import find_indices

nested_list = [[2, 5], [10, 20, 30],[11,12,13,14]]
tensor = list(range(100))

indices = find_indices.find_indices(tensor, nested_list)

print(indices)  

# prints [[2, 5], [10, 20, 30]]
