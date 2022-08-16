class Person(object):
    # 封装，一种权限的控制。 把一些重要的隐私的数据，不想被外界直接访问或者修改的数据，进行保护。
    def __init__(self, name, age):
        self.name = name # 实例变量，成员变量。
        self.age = age
        self.__life_val = 100 # 私有变量，私有属性。

    def get_life_val(self):
        print("生命值还有", self.__life_val)
        return self.__life_val

    def got_attack(self,n):
        self.__life_val -= n
        print("被攻击了%s滴血。" % n)
        self.__breath()
        return self.__life_val

    def __breath(self): # 私有方法
        print("%s在呼吸" % self.name)

a = Person("WBOY", 22)
#print(a.__life_val)
a.get_life_val()
a.got_attack(20)

a._Person__breath() # 实例名._类名+方法名，就能够访问私有方法。


a._Person__life_val = 10 # 修改私有属性。
a.get_life_val()

