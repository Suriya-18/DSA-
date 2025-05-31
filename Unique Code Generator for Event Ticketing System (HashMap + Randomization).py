import random
import string

class TicketSystem:
    def __init__(self):
        self.user_to_code = {}
        self.used_codes = set()

    def generate_unique_code(self, length=8):
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            if code not in self.used_codes:
                self.used_codes.add(code)
                return code

    def register_user(self, user_id):
        if user_id in self.user_to_code:
            return self.user_to_code[user_id]
        code = self.generate_unique_code()
        self.user_to_code[user_id] = code
        return code

    def get_user_code(self, user_id):
        return self.user_to_code.get(user_id, None)

# Usage
system = TicketSystem()
print(system.register_user("user123"))  # Unique ticket code
print(system.register_user("user123"))  # Same code as above
print(system.get_user_code("user123"))
