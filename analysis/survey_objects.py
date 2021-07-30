class style_data:
    def __init__(self, rating, variance, std_dev):
        self.rating = rating
        self.variance = variance
        self.std_dev = std_dev

    def __str__(self):
        return f'rating: {self.rating:.2f} variance: {self.variance:.2f} std_dev: {self.std_dev:.2f}'