"""
This module represents the Consumer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Class that represents a consumer.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor.

        :type carts: List
        :param carts: a list of add and remove operations

        :type marketplace: Marketplace
        :param marketplace: a reference to the marketplace

        :type retry_wait_time: Time
        :param retry_wait_time: the number of seconds that a producer must wait
        until the Marketplace becomes available

        :type kwargs:
        :param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.mk_p = marketplace
        self.ops = {"add": self.mk_p.add_to_cart,
                    "remove": self.mk_p.remove_from_cart}
        self.wait = retry_wait_time


    def run(self):
        for cart in self.carts:
            _id = self.mk_p.new_cart()

            for op_in_cart in cart:
                no_of_op = 0
                while no_of_op < op_in_cart["quantity"]:
                    result = self.ops[op_in_cart["type"]](_id, op_in_cart["product"])

                    if result is None:
                        no_of_op += 1
                    elif result is True:
                        no_of_op += 1
                    else:
                        time.sleep(self.wait)

            self.mk_p.place_order(_id)
