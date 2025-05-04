"""
UUID, custom alphanumeric codes, hashing, and timestamp-based methods with random components"""
import random
import string
import hashlib
import uuid
import time


# 1. UUID-based Project Code (Globally Unique)
def generate_uuid_project_code():
    """Generate a globally unique project code using UUID."""
    return str(uuid.uuid4())


# 2. Custom Alphanumeric Code with Prefix, Timestamp, and Random Suffix
def generate_custom_project_code(prefix="PRJ", length=12):
    """Generate a unique project code with a custom prefix and random suffix."""
    timestamp = int(time.time())  # Timestamp-based unique element
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length-8))
    return f"{prefix}-{timestamp}-{random_suffix}"


# 3. Hash-based Project Code (Consistent Unique Code based on input)
def generate_hash_project_code(input_string):
    """Generate a unique project code by hashing an input string."""
    hashed = hashlib.sha256(input_string.encode()).hexdigest()[:16]  # Take the first 16 characters
    return hashed


# 4. Timestamp + Counter + Random Code for Readable Project Codes
def generate_sequential_project_code(counter=1):
    """Generate a readable project code with timestamp, counter, and random part."""
    timestamp = int(time.time())  # Current time in seconds
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{timestamp}-{counter}-{random_part}"


# Main Function to Test All Methods
def generate_project_codes(project_name="BigDataProject123", counter=1):
    print("1. UUID-based Project Code:")
    uuid_code = generate_uuid_project_code()
    print(f"  {uuid_code}\n")
    
    print("2. Custom Alphanumeric Project Code:")
    custom_code = generate_custom_project_code()
    print(f"  {custom_code}\n")
    
    print("3. Hash-based Project Code:")
    hash_code = generate_hash_project_code(project_name)
    print(f"  {hash_code}\n")
    
    print("4. Timestamp + Counter + Random Code:")
    sequential_code = generate_sequential_project_code(counter)
    print(f"  {sequential_code}\n")


# Example Usage
if __name__ == "__main__":
    generate_project_codes(project_name="BigDataProject123", counter=101)
