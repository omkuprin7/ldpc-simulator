import random
import math


class Generator:
    @staticmethod
    def generate_bit_sequence(size):
        """Generate a random bit sequence of given size."""
        return [random.randint(0, 1) for _ in range(size)]

    def __init__(self, idum, sigma):
        self.idum = idum
        self.sigma = sigma

    def ran(self):
        """Linear congruential generator."""
        k = self.idum // 127773
        self.idum = 16807 * (self.idum - k * 127773) - 2836 * k
        if self.idum < 0:
            self.idum += 2147483647
        ans = (1.0 / 2147483647) * self.idum
        return ans

    def gauss(self, b):
        """Generate Gaussian random number using Box-Muller transform."""
        magnitude = self.sigma * math.sqrt(-2.0 * math.log(self.ran()))
        angle = 2.0 * math.pi * self.ran()

        if b % 2 == 0:
            return magnitude * math.cos(angle)
        else:
            return magnitude * math.sin(angle)
