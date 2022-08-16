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

class Person:
    role = 'person'
    def __init__(self, name, attack_val, sex):
        self.name = name
        self.attack_val = attack_val
        self.sex = sex
        self.life_val = 100

    def attack(self, dog):
        dog.life_val -= self.attack_val
        print("人[%s]打了狗[%s]一棒，狗掉了[%s]点血，还有[%s]血"
              % (self.name,
                 dog.name,
                 self.attack_val,
                 dog.life_val))




p1 = Person('wboy', 50, 'M')
d1 = Dog('wang cai', '金毛', 30, p1)

d1.say_hi()