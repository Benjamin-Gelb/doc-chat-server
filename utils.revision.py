import random
import string

def generate_unique_char_string(length=8):
    """ Generate a random string of fixed length """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))