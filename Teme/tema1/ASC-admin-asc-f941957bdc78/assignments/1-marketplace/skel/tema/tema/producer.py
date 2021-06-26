"""
This module represents the Producer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread
import time

class Producer(Thread):
    """
    Class that represents a producer.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.mk_p = marketplace
        self.wait = republish_wait_time

        self._id = self.mk_p.register_producer()

    def run(self):
        while True:
            for (prod, quant, wait) in self.products:
                i = 0
                while i < quant:
                    ret = self.mk_p.publish(str(self._id), prod)

                    if ret:
                        time.sleep(wait)
                        i += 1
                    else:
                        time.sleep(self.wait)
