import math
import random
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
from generator import Generator
from constants import IDUM1, IDUM2


class Channel:
    def __init__(self, mode, p, mod, L_c1, L_c2, L_c3):
        self.mode = mode
        self.p = p
        self.modulation = mod
        self.L_c1 = L_c1
        self.L_c2 = L_c2
        self.L_c3 = L_c3
        self.gen_ptr = None
        self.gen_ptr2 = None
        # Random number generator for mode 1 (AWGN)
        # Note: In C++ code, a new generator is created for each bit (inefficient)
        # We use a single generator per channel instance for better performance
        # The generator is initialized with a random seed for each channel instance
        if NUMPY_AVAILABLE:
            # Use numpy's random generator with a random seed for reproducibility
            # Each channel instance gets its own independent RNG
            import time
            self._rng = np.random.RandomState(int(time.time() * 1000000) % (2**31))
        else:
            self._rng = None

    def __del__(self):
        """Destructor - cleanup generators."""
        pass  # Python handles garbage collection

    def process(self, data_buffer):
        """Process data through the channel."""
        for i in range(len(data_buffer._encoded_data)):
            bit = 0.0
            pom1 = 0.0
            pom2 = 0.0
            a = 0.0
            b = 0.0
            
            # ФМ-2 модуляция
            if self.modulation == 1:
                bit = -1.0 if (data_buffer._encoded_data[i] == 0) else 1.0
            elif self.modulation == 2:  # ФМ-4 модуляция
                bit = -0.7 if (data_buffer._encoded_data[i] == 0) else 0.7

            # выбор типа помехи
            if self.mode == 1:  # 1 генератор (АБГШ)
                # ВАЖНО: В C++ коде используется:
                # std::normal_distribution<double> distribution(0.0, pow(gen_ptr->sigma, 2));
                # std::normal_distribution принимает (mean, stddev), а не (mean, variance)
                # 
                # В C++ коде есть ОШИБКА - передается pow(sigma, 2) как stddev
                # Это означает, что фактически генерируется шум с stddev = sigma^2
                # что соответствует variance = sigma^4 (а не sigma^2)
                #
                # Чтобы соответствовать C++ поведению, мы должны использовать ту же ошибку:
                # Генерируем шум с stddev = sigma^2 (как в C++ коде)
                if NUMPY_AVAILABLE:
                    if self._rng is not None:
                        # Соответствуем C++ коду: используем sigma^2 как stddev
                        pom1 = self._rng.normal(0.0, self.gen_ptr.sigma ** 2)
                    else:
                        pom1 = np.random.normal(0.0, self.gen_ptr.sigma ** 2)
                else:
                    pom1 = random.gauss(0.0, self.gen_ptr.sigma ** 2)

                # Принятый сигнал: y = bit + noise
                # где bit ∈ {-1, 1} для BPSK
                yn = bit + pom1
                
                # LLR (Log-Likelihood Ratio) - соответствует C++ коду
                # В C++: dLLR = 2 * yn / pow(gen_ptr->sigma, 2)
                d_llr = 2.0 * yn / (self.gen_ptr.sigma ** 2)
                data_buffer._channel_data.append(d_llr)

            if self.mode == 2:  # АБГШ + помеха в части полосы
                # В C++ используется rand() % _dataBuffer->_encodedData.size()
                if NUMPY_AVAILABLE:
                    a = np.random.randint(0, len(data_buffer._encoded_data))
                else:
                    a = random.randint(0, len(data_buffer._encoded_data) - 1)
                b = a / len(data_buffer._encoded_data)
                pom1 = self.gen_ptr.gauss(i)
                if b < self.p:
                    pom2 = self.gen_ptr2.gauss(i)
                    data_buffer._channel_data.append((bit + pom2 + pom1) * self.L_c2)
                else:
                    data_buffer._channel_data.append((bit + pom1) * self.L_c1)

            if self.mode == 3:  # АБГШ + встречная помеха
                pom1 = self.gen_ptr.gauss(i)
                pom2 = self.gen_ptr2.gauss(i)
                data_buffer._channel_data.append(((bit + pom1 + pom2) * self.p + (bit + pom1) * (1 - self.p)) * self.L_c3)

    @staticmethod
    def create_channel(speed, sn1, sn2, mode, p, mod):
        """Create a channel with specified parameters."""
        L_c1 = 4.0 * speed * (10.0 ** (sn1 * 0.1))
        L_c2 = 4.0 * speed / ((1.0 / (10.0 ** (sn1 * 0.1))) + (1.0 / ((10.0 ** (sn2 * 0.1)) * p)))
        L_c3 = 4.0 * p * speed / (1.0 / (10.0 ** (sn2 * 0.1)) + 1.0 / (10.0 ** (sn2 * 0.1))) + 4 * speed * (1 - p) * (10.0 ** (sn1 * 0.1))

        sigma1 = 0.0
        sigma2 = 0.0

        if mode == 1:
            sigma1 = 1.0 / math.sqrt(2.0 * speed * (10.0 ** (sn1 * 0.1)))
        if mode == 2:
            sigma1 = 1.0 / math.sqrt(2.0 * speed * (10.0 ** (sn1 * 0.1)))
            sigma2 = 1.0 / math.sqrt(2.0 * speed * ((10.0 ** (sn2 * 0.1)) * p))
        if mode == 3:
            sigma1 = 1.0 / math.sqrt(2.0 * speed * (10.0 ** (sn1 * 0.1)))
            sigma2 = 1.0 / math.sqrt(2.0 * speed * (10.0 ** (sn2 * 0.1)))

        channel = Channel(mode, p, mod, L_c1, L_c2, L_c3)
        channel.gen_ptr = Generator(IDUM1, sigma1)
        channel.gen_ptr2 = Generator(IDUM2, sigma2)

        return channel
