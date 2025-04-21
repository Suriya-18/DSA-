"""
Problem: Design a Rate Limiter
Problem Statement:
Design a rate limiter that limits the number of requests a user can make to a service in a given time window (for example, 100 requests per minute). The rate limiter should allow you to:

Record a request for a specific user at a specific time.

Check if a user’s request exceeds the rate limit for the time window.

Constraints:
You need to handle many users.

The system should be able to handle multiple requests in a second.

Assume we need to handle time windows of 1 minute, 1 hour, etc.

We should return True for a request if it’s within the rate limit, and False if it exceeds the rate limit.""""

import collections
import time

class RateLimiter:
    def __init__(self, limit: int, window: int):
        """
        Initialize the rate limiter with a limit and a time window.
        :param limit: maximum number of requests allowed in the time window.
        :param window: time window in seconds (e.g., 60 seconds for 1 minute).
        """
        self.limit = limit  # maximum number of requests allowed
        self.window = window  # time window in seconds
        self.requests = collections.defaultdict(collections.deque)  # stores user requests with timestamps
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if the user is allowed to make the request.
        :param user_id: unique identifier for the user.
        :return: True if allowed, False if rate limit exceeded.
        """
        current_time = int(time.time())  # get current timestamp in seconds
        user_requests = self.requests[user_id]
        
        # Remove requests that are outside the time window (older than current_time - window)
        while user_requests and user_requests[0] <= current_time - self.window:
            user_requests.popleft()
        
        # If the number of requests in the window exceeds the limit, deny the request
        if len(user_requests) >= self.limit:
            return False
        
        # Otherwise, allow the request and add the current time to the deque
        user_requests.append(current_time)
        return True
    
    def get_request_count(self, user_id: str) -> int:
        """
        Get the count of requests the user has made in the current time window.
        :param user_id: unique identifier for the user.
        :return: count of requests made in the time window.
        """
        current_time = int(time.time())
        user_requests = self.requests[user_id]
        
        # Remove requests that are outside the time window (older than current_time - window)
        while user_requests and user_requests[0] <= current_time - self.window:
            user_requests.popleft()
        
        return len(user_requests)

# Example usage:
if __name__ == "__main__":
    limiter = RateLimiter(5, 60)  # Limit 5 requests per 60 seconds for each user
    
    user_id = "user1"
    
    # Simulating requests for user1
    print(limiter.is_allowed(user_id))  # True
    print(limiter.is_allowed(user_id))  # True
    print(limiter.is_allowed(user_id))  # True
    print(limiter.is_allowed(user_id))  # True
    print(limiter.is_allowed(user_id))  # True
    print(limiter.is_allowed(user_id))  # False (rate limit exceeded)
    
    # Simulate passing of time, then another request
    time.sleep(60)  # Wait for 60 seconds (1 minute)
    print(limiter.is_allowed(user_id))  # True (after waiting 1 minute)
