class Dog(object):
    def sound(self):
        print(" wang ...")

class Cat(object):
    def sound(self):
        print(" miao ...")

def make_sound(animal_obj):
    """统一调用接口"""
    animal_obj.sound()

d = Dog()
c = Cat()

make_sound(d)

make_sound(c)