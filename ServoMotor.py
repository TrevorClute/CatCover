import pigpio
import time
import asyncio


CLOSE_GPIO = 13
OPEN_GPIO = 12


class ServoMotor:
    pi = pigpio.pi()
    is_open = True


    def __init__(self):
        self.pi.set_mode(CLOSE_GPIO, pigpio.OUTPUT)
        self.pi.set_mode(OPEN_GPIO, pigpio.OUTPUT)

    def close(self):
        # if not self.is_open:
        #     return
        self.pi.set_servo_pulsewidth(CLOSE_GPIO, 1200)
        self.pi.set_servo_pulsewidth(OPEN_GPIO, 1600)
        time.sleep(0.3)
        self.pi.set_servo_pulsewidth(CLOSE_GPIO, 0)
        self.pi.set_servo_pulsewidth(OPEN_GPIO, 0)
        self.is_open = False

    def open(self):
        # if self.is_open:
        #     return
        self.pi.set_servo_pulsewidth(OPEN_GPIO, 1385)
        self.pi.set_servo_pulsewidth(CLOSE_GPIO, 1490)
        time.sleep(1.1)
        self.pi.set_servo_pulsewidth(OPEN_GPIO, 0)
        self.pi.set_servo_pulsewidth(CLOSE_GPIO, 0)
        self.is_open = True





# pi.set_servo_pulsewidth(WHITE_GPIO, 1470)
# pi.set_servo_pulsewidth(BLACK_GPIO, 1460)
# time.sleep(1)


# pi.set_servo_pulsewidth(WHITE_GPIO, 0)
# pi.set_servo_pulsewidth(BLACK_GPIO, 0)
# pi.stop()

# black
# first forward = 1480
# center = 1460
# first backward = 1410
# max = 1900
#
# white
# first forward = 1500
# center = 1460
#jfirst backward = 1435
