class Animal:
    # 有共性。
    a_type = "哺乳动物"

    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex

    def eat(self):
        print("%s is eating .." % self.name)


class Person(Animal):  # 加括号就是继承了括号里面的东西。
    a_type = "人类"  # 重写

    def __init__(self, name, age, sex, hobby):  # 父类构造方法优先。
        # Animal.__init__(self,name,age,sex)
        super(Person, self).__init__(name, age, sex)  # 与 "Animal.__init__(self,name,age,sex)" 的功能相同
        self.hobby = hobby

    def talk(self):
        print("%s is talking" % self.name)

    def eat(self):  # 重写。
        # Animal.eat(self)
        super().eat()
        print("%s is eating on the table." % self.name)


class Dog(Animal):  # 加括号就是继承了括号里面的东西。
    a_type = "狗"  # 重写。

    def hunt(self):
        print("%s is hunting..." % self.name)


p = Person("wboy", 24, "M", "MUSIC")
p.eat()
p.talk()
print(p.a_type)
print(p.name, p.hobby)
d = Dog("wangcai", 1, "F")
d.eat()
d.hunt()
print(d.a_type)
