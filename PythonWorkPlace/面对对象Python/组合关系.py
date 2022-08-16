
# 组合关系，由一堆组件构成一个完整的实体，组件本身独立，但是又不能自己运行。必须跟宿主组合在一起运行。

class Dog:
    # 属性，类属性，类变量
    role = 'dog'

    # 初始化方法，构造方法，构造函数， 实例化时会自动执行，进行一些初始化工作。
    def __init__(self, name, d_type, attack_val, master):
        self.name = name
        self.d_type = d_type
        self.attack_val = attack_val
        self.life_val = 100
        self.master = master

    def bite(self, person):
        person.life_val -= self.attack_val
        print("狗[%s]咬了[%s]一口，[%s]收到了[%s]点伤害，还有血量[%s].."
              % (self.name,
                 person.name,
                 person.name,
                 self.attack_val,
                 person.life_val))

    def say_hi(self):
        print("Hi，I am %s, My master is %s"
              %(self.name, self.master.name))

class Weapon:

    def dag_stick(self, obj):
        """打狗棒"""
        self.name = "打狗棒"
        self.attack_val = 40
        obj.life_val -= self.attack_val
        self.print_log(obj)

class Person:
    role = 'person'
    def __init__(self, name, attack_val, sex):
        self.name = name
        #self.attack_val = attack_val
        self.sex = sex
        self.life_val = 100
        self.weapon = Weapon() # 直接进行实例化

    def attack(self, dog):
        dog.life_val -= self.attack_val
        print("人[%s]打了狗[%s]一棒，狗掉了[%s]点血，还有[%s]血"
              % (self.name,
                 dog.name,
                 self.attack_val,
                 dog.life_val))



p1 = Person("WBOY", 40, "M")
d1 = Dog("wangcai", "金毛", 30, p1)

d1.bite(p1)



p1.weapon.dag_stick(d1)
