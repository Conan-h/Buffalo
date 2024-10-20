import remove_values

my_list = [1, 2, 3, 2, 4, 2, 5]
value_to_remove = [2,3]

new_list = remove_values.remove_values(my_list, value_to_remove)

print(new_list)  # Outputs: [1, 3, 4, 5]