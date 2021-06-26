"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2021
"""
from threading import Lock, currentThread

class Marketplace:
    """
    Class that represents the Marketplace. It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """
    lock_reg = Lock()
    lock_carts = Lock()
    lock_alter = Lock()
    print = Lock()
    no_carts = None # numar total de carturi
    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue associated with each producer
        """
        self.no_carts = 0
        self.max_prod_q_size = queue_size_per_producer
        self.prods = []  # lista cu toate produsele
        self.carts = {}  # dictionar cu id-cart
        self.producers = {}  # mapare producator-produs
        self.prod_q_sizes = []  # numarul de produse ale unui producator

    def register_producer(self):
        """
        Returns an id for the producer that calls this.
        """
        with self.lock_reg:
            _id = len(self.prod_q_sizes)
            self.prod_q_sizes.append(0)

        return _id

    def publish(self, producer_id, product):
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False, it should wait and then try again.
        """
        _id = int(producer_id)

        if self.prod_q_sizes[_id] >= self.max_prod_q_size:
            return False

        self.prod_q_sizes[_id] += 1
        self.prods.append(product)
        self.producers[product] = _id

        return True

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """
        with self.lock_carts:
            self.no_carts += 1
            cart_id = self.no_carts

        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False, it should wait and then try again
        """
        with self.lock_alter:
            if product not in self.prods:
                return False

            self.prod_q_sizes[self.producers[product]] -= 1
            self.prods.remove(product)

        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """
        self.carts[cart_id].remove(product)
        self.prods.append(product)

        with self.lock_alter:
            self.prod_q_sizes[self.producers[product]] += 1


    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        prod_list = self.carts.pop(cart_id, None)

        for product in prod_list:
            with self.print:
                print(currentThread().getName(), "bought", product)

        return prod_list
