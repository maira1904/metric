# Sample Problem Solution in Python

def sum_of_evens(numbers):
    even_sum = 0
    for num in numbers:
        if num % 2 == 0:
            even_sum += num
    return even_sum

# Example usage
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = sum_of_evens(numbers)
print("Sum of even numbers:", result)
