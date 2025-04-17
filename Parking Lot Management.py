"""
A parking lot has limited space. When a car enters, it's registered with its number. When it exits, its duration is calculated.

Problem: Design a system that records car entry and exit and calculates the total parking duration for each car.
"""

import time
from collections import defaultdict

class ParkingLot:
    def __init__(self):
        self.entry_time = {}  # car_number -> entry_timestamp
        self.parking_duration = defaultdict(int)  # car_number -> total_seconds

    def enter(self, car_number):
        if car_number in self.entry_time:
            print(f"🚫 Car {car_number} is already inside the lot!")
        else:
            self.entry_time[car_number] = time.time()
            print(f"✅ Car {car_number} entered at {time.ctime(self.entry_time[car_number])}")

    def exit(self, car_number):
        if car_number not in self.entry_time:
            print(f"🚫 Car {car_number} is not in the lot!")
        else:
            exit_time = time.time()
            duration = exit_time - self.entry_time[car_number]
            self.parking_duration[car_number] += duration
            del self.entry_time[car_number]
            print(f"✅ Car {car_number} exited at {time.ctime(exit_time)}")
            print(f"🕒 Duration: {int(duration)} seconds")

    def total_duration(self, car_number):
        total = self.parking_duration[car_number]
        print(f"⏳ Total parking duration for {car_number}: {int(total)} seconds")

    def show_parked_cars(self):
        if not self.entry_time:
            print("🅿️ Parking lot is empty!")
        else:
            print("🚗 Cars currently in the lot:")
            for car in self.entry_time:
                print(f" - {car} (entered at {time.ctime(self.entry_time[car])})")

# -----------------------------
# 💻 Example usage:
# -----------------------------
if __name__ == "__main__":
    lot = ParkingLot()

    lot.enter("TN09AB1234")
    time.sleep(2)

    lot.enter("TN22CD5678")
    time.sleep(3)

    lot.exit("TN09AB1234")
    time.sleep(1)

    lot.show_parked_cars()

    lot.exit("TN22CD5678")

    lot.total_duration("TN09AB1234")
    lot.total_duration("TN22CD5678")
